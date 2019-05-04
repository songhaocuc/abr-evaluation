from evaluation_server.bottle import route, run, request, response
import evaluator.evaluator as evaluator
import json


@route('/hello')
def hello():
    return "Hello world!"


@route('/evaluation', method='POST')
def evaluate():
    args = request.json
    result = evaluator.evaluate(args)
    return json.dumps(result)


run(host='localhost', port=3001, debug=True)
