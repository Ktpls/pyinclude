import flask


class MiniumSimpleServer:
    def __init__(self, port):
        self.port = port
        self.app = flask.Flask(__name__)

    def run(self):
        self.app.run(host="0.0.0.0", port=self.port, debug=True, reload=False)
