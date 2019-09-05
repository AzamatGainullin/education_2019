from flask import Flask, render_template, request, redirect,  url_for, flash
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

if __name__ == "__main__":
    app.run()