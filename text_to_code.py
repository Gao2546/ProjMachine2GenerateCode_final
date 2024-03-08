from flask import Flask, render_template,request,redirect,url_for,jsonify
from flask_wtf import FlaskForm # 1.
from wtforms import StringField # 2.
from wtforms.validators import DataRequired # 3.
import model_text_to_code

class SimpleForm(FlaskForm): # 1.
	name = StringField(label="question", validators=[DataRequired()])

app = Flask(__name__)
app.secret_key = "secret"

@app.route("/",methods = ['GET','POST'])
def index():
    simple_form = SimpleForm(request.form)
    return render_template("index.html", form=simple_form,len = len(list_user_massage),list_user_massage = list_user_massage,list_system_massage = list_system_massage)

@app.route('/code/', methods=['post'])
def code():
    simple_form = SimpleForm()
    output = ""
    if request.method == "POST":
        question = simple_form.name.data
        simple_form.name.data = ""
        list_user_massage.append(question)
        output = model_text_to_code.gencode(question)
        list_system_massage.append(output)
        # Do something with name
    else:
        print("Name is required")
        # OR send some error message to front-end
    if request.method == "POST":
        return jsonify(data={"user-message":list_user_massage,"system-message":list_system_massage})
    return jsonify(data=simple_form.errors)
    
if __name__ == "__main__":
    list_user_massage = []
    list_system_massage = []
    app.run(debug=True,port=5000)