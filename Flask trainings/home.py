from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello Flask'

@app.route('/user/<int:user_id>/')
def user_profile(user_id):
    return "Profile page of user #{}".format(user_id)

@app.route('/books/<genre>')
def books(genre):
    res = make_response("All Books in {} category".format(genre))
    res.headers['Content-Type'] = 'text/plain'
    res.headers['Server'] = 'Foobar'
    return res


@app.route('/r')
def requestdata():
    return "Hello! Your IP is {} and you are using {}: ".format(request.remote_addr,  
                                                                request.user_agent)
@app.route('/set-cookie')
def set_cookie():
    res = make_response("Cookie setter")
    res.set_cookie("favorite-color", "skyblue", 60*60*24*15)
    res.set_cookie("favorite-font", "sans-serif", 60*60*24*15)
    return res

if __name__ == "__main__":
    app.run(debug=True)