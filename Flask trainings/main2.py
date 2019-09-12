from flask import Flask, render_template, request, redirect,  url_for, flash, make_response
from flask_script import Manager, Command, Shell
from forms import ContactForm
import pandas as pd

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'a really really really really long secret key'

@app.route('/')
def index():
    name, age, profession = "Jerry", 24, 'Programmer'
    template_context = dict(name=name, age=age, profession=profession)
    return render_template('index.html', **template_context)

@app.route('/sample')
def index1():
    return render_template('index2.html')

@app.route('/login/', methods=['post', 'get'])
def login():
    message = ''
    username = ''
    password = ''
    if request.method == 'POST':
        username = request.form.get('username')  # запрос к данным формы
        password = request.form.get('password')

    if username == 'root' and password == 'pass':
        message = "Correct username and password"
    else:
        message = "Wrong username or password"

    return render_template('login.html', message=message)

@app.route('/contact/', methods=['get', 'post'])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        message = form.message.data
        passwordfield = form.passwordfield.data
        print(name)
        print(email)
        print(message)
        print(passwordfield)
        df = pd.DataFrame([name, email, message, passwordfield])
        df.to_csv('df.txt')
        # здесь логика базы данных
        print("\nData received. Now redirecting ...")
        flash("Message Received", "success")
        return redirect(url_for('contact'))

    return render_template('contact.html', form=form)
   
@app.route('/cookie/')
def cookie():
    if not request.cookies.get('foo'):
        res = make_response("Setting a cookie")
        res.set_cookie('foo', 'bar', max_age=60*60*24*365*2)
    else:
        res = make_response("Value of cookie foo is {}".format(request.cookies.get('foo')))
    return res

@app.route('/delete-cookie/')
def delete_cookie():
    res = make_response("Cookie Removed")
    res.set_cookie('foo', 'bar', max_age=0)
    return res

@app.route('/article/', methods=['POST',  'GET'])
def article():
    if request.method == 'POST':
        print(request.form)
        res = make_response("")
        res.set_cookie("font", request.form.get('font'), 60*60*24*15)
        res.headers['location'] = url_for('article')
        return res, 302

    return render_template('article.html')    

if __name__ == "__main__":
    app.run()