# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Replay stored trajectories and simulate brax systems."""

from wsgiref import simple_server
from wsgiref import validate

from absl import app
from absl import flags
from brax.base import Actuator, Convex, Mesh, Plane, Transform
from brax.generalized import pipeline as generalized
from brax.io import html
from brax.io import mjcf
from brax.positional import pipeline as positional
from brax.spring import pipeline as spring
from etils import epath
import flask
from flask import jsonify
from flask import request
from flask import send_from_directory
import flask_cors
import jax
from jax import numpy as jp


PORT = flags.DEFINE_integer(
    name='port', default=8080, help='Port to run server on'
)
DEBUG = flags.DEFINE_boolean(
    name='debug', default=False, help='Debug the server.'
)

flask_app = flask.Flask(__name__)
flask_cors.CORS(flask_app)


@flask_app.route('/', methods=['GET'])
def index():
  return jsonify(success=True)


@flask_app.route('/favicon.ico')
def favicon():
  """Serves the brax favicon."""
  path = epath.Path(flask_app.root_path)
  return send_from_directory(str(path), 'favicon.ico')


@flask_app.route('/js/<path:path>', methods=['GET'])
def js(path):
  """Serves files from the js/ directory."""
  path = epath.Path(flask_app.root_path) / 'js' / path
  response = flask.Response(path.read_text(), mimetype='text/javascript')
  response.headers.add('Access-Control-Allow-Origin', '*')
  return response


@flask_app.route('/play/<path:path>', methods=['GET'])
def play_trajectory(path):
  """Renders a json-encoded brax trajectory from a local file path."""
  system = epath.Path(path).read_text()
  return html.render_from_json(
      system, height='100vh', colab=False, base_url='/js/viewer.js'
  )


@flask_app.route('/simulate/<path:path>', methods=['GET'])
def simulate(path):
  """Simulates a brax system from a local file path."""
  sys = mjcf.load(path)
  pipeline = {
      'generalized': generalized,
      'positional': positional,
      'spring': spring,
  }[request.args.get('pipeline', 'generalized')]
  steps = int(request.args.get('steps', 1000))
  act_fn = request.args.get('act', 'sin')
  solver_iterations = int(request.args.get('solver_iterations', 100))
  force_floor = request.args.get('force_floor', 'false').lower() == 'true'
  add_floor = request.args.get('add_floor', 'false').lower() == 'true'
  add_act = request.args.get('add_act', 'false').lower() == 'true'

  is_mesh = lambda g: isinstance(g, Mesh) and not isinstance(g, Convex)

  if force_floor:
    floors = [g for g in sys.geoms if isinstance(g, Plane)]
    if floors:
      floor = floors[0]
      masks = [
          0 if is_mesh(g) else 1 if g is floor else 1 << 32 for g in sys.geoms
      ]
      sys = sys.replace(geom_masks=masks)
  if solver_iterations > 0:
    sys = sys.replace(solver_iterations=solver_iterations)
  if add_floor and not [g for g in sys.geoms if isinstance(g, Plane)]:
    geoms = sys.geoms + [
        Plane(
            link_idx=None,
            transform=Transform.zero((1,)),
            friction=jp.ones((1,)),
            elasticity=jp.ones((1,)),
        )
    ]
    geom_masks = [0 if is_mesh(g) else 1 for g in sys.geoms]
    geom_masks.append(1 << 32 | 1)
    sys = sys.replace(geoms=geoms, geom_masks=geom_masks)
  if add_act and not sys.actuator_types:
    # some configs (like urdfs) have no actuators
    actuator_types = ''.join(['m' for t in sys.link_types if t in '123'])
    actuator_link_id = [i for i, t in enumerate(sys.link_types) if t in '123']
    actuator_qid = [int(i) for i in sys.q_idx('123')]
    actuator_qdid = [int(i) for i in sys.qd_idx('123')]
    actuator = Actuator(
        ctrl_range=jp.tile(jp.array([-1.0, 1.0]), (len(actuator_types), 2)),
        gear=25 * jp.ones(len(actuator_types)),
    )
    sys = sys.replace(
        actuator=actuator,
        actuator_types=actuator_types,
        actuator_link_id=actuator_link_id,
        actuator_qid=actuator_qid,
        actuator_qdid=actuator_qdid,
    )

  jit_init, jit_step = jax.jit(pipeline.init), jax.jit(pipeline.step)
  states = [jit_init(sys, sys.init_q, jp.zeros(sys.qd_size()))]
  for i in range(steps):
    if act_fn == 'sin':
      act = 0.5 * jp.sin(jp.ones(sys.act_size()) * 5 * i * sys.dt)
    elif act_fn == 'zero':
      act = jp.zeros(sys.qd_size())
    elif act_fn == 'zero_p':
      q = states[-1].q[sys.q_idx('123')]
      act = -q
    states.append(jit_step(sys, states[-1], act))
  return html.render(
      sys, states, height='100vh', colab=False, base_url='/js/viewer.js'
  )


def main(_):
  if DEBUG.value:
    flask_app.run(
        host='localhost', port=PORT.value, use_reloader=True, debug=True
    )
    return

  server = simple_server.make_server(
      'localhost', PORT.value, validate.validator(flask_app)
  )
  server.serve_forever()


if __name__ == '__main__':
  app.run(main)
