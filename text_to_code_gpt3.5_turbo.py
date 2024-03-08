from flask import Flask, render_template,request,redirect,url_for,jsonify
from flask_wtf import FlaskForm # 1.
from wtforms import StringField # 2.
from wtforms.validators import DataRequired # 3.
import model_text_to_code
from openai import OpenAI
import re
client = OpenAI()

class SimpleForm(FlaskForm): # 1.
	name = StringField(label="question", validators=[DataRequired()])

app = Flask(__name__)
app.secret_key = "secret"

@app.route("/", methods=["GET", "POST"])
def index():
    simple_form = SimpleForm(request.form)
    return render_template("new_index.html", form=simple_form,len = len(list_user_massage),list_user_massage = list_user_massage,list_system_massage = list_system_massage)

@app.route('/code/', methods=['POST'])
def code():
    simple_form = SimpleForm()
    output = ""
    numseq = []
    seq_text = []
    st_with_code = False
    if request.method == "POST":
        question = simple_form.name.data
        simple_form.name.data = ""
        list_user_massage.append(question)
        api_message.append({"role":"user","content":question})
        #output = model_text_to_code.gencode(question)
        completion = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=api_message,
              max_tokens=500
            )
        output = completion.choices[0].message.content
        # output = 'Sure! Here is a simple example of an HTML code:\n\n```html\n<!DOCTYPE html>\n<html>\n<head>\n  <title>Sample Page</title>\n</head>\n<body>\n  <h1>Welcome to my Sample Page</h1>\n  <p>This is a paragraph with some sample text.</p>\n  <a href="https://www.example.com">Click here for more information</a>\n</body>\n</html>\n```\n\nThis code creates a basic HTML page with a heading, paragraph, and a hyperlink. Save it as an .html file and open it in a web browser to see how it looks. Let me know if you need any more assistance!'.strip("\n")

        print([output])
        
        if output[:3] == "```":
            st_with_code = True
        #api_message.append({"role":"system","content":output})
        api_code = re.findall("```.+```",output,re.DOTALL)
        print(api_code)
        count = 0
        for reg in api_code:
            output = output.replace(reg,f"<SUBSEQ{count}>")
            numseq.append(f"<SUBSEQ{count}>")
            count +=1
        for i,j in enumerate(numseq):
            if st_with_code:
                seq_text.append("code")
                seq_text.append(api_code[i].strip("```").strip("\n"))
                if (output.split(j)[0].strip("\n") != ""):
                    seq_text.append("text")
                    seq_text.append(output.split(j)[0].strip("\n"))
                output = output.replace(seq_text[-1],"",1).replace(j,"",1)
            else:
                seq_text.append("text")
                seq_text.append(output.split(j)[0].strip("\n"))
                output = output.replace(seq_text[-1],"",1).replace(j,"",1)
                if api_code[i].strip("```").strip("\n") != "":
                    seq_text.append("code")
                    seq_text.append(api_code[i].strip("```").strip("\n"))
        if len(output) > 1:
            if st_with_code:
                seq_text.append("code") 
                seq_text.append(output.strip("\n"))
            else:
                seq_text.append("text") 
                seq_text.append(output.strip("\n"))
        if len(seq_text) == 0:
             seq_text.append(output)
        list_system_massage.append(seq_text)
        # Do something with name
    else:
        print("Name is required")
        # OR send some error message to front-end
    if request.method == "POST":
        return jsonify(data={"user-message":list_user_massage,"system-message":list_system_massage})
    return jsonify(data=simple_form.errors)
    
if __name__ == "__main__":
    api_message = [{'role':"user",'content':"you are expert python code"},{'role':"user",'content':"give me only python code no explain after this prompt"}]
    api_code = []
    api_text = []
    list_user_massage = []
    list_system_massage = []
    app.run(debug=True,port=5000)