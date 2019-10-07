from flask import Flask, request, render_template, flash
from flask_sqlalchemy import SQLAlchemy
import time
import random
import cachetools
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
app = Flask(__name__)
app.config.update(
   DEBUG = True,
   SECRET_KEY = 'asdfsdfssf asf dsgsdg',
   WTF_CSRF_ENABLED = False
)
def get_data():
   #time.sleep(10)
   return str(random.randrange(1, 100))
DF = get_data()
def update_DF():
   global DF
   DF = get_data()
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_DF, trigger='interval', seconds=20)
scheduler.start()

@app.route('/')
def index():
   report_data = DF
   return report_data
atexit.register(lambda: scheduler.shutdown())
if __name__ == '__main__':
   app.run(host='localhost', port=5000)