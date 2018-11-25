import io
from flask import render_template, flash, redirect, url_for, make_response, send_file, request
# from app import app, db
from app import app
from src import data
from flask_login import current_user, login_user, logout_user, login_required


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title="Home")

cfpb = data.CFPBdata().get_data()

@app.route('/cfpb_data)
def cfpb_data():
    return f"The total number of CFPB complaints is {len(cfpb)}."

