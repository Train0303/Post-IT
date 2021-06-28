from enum import Flag
from flask import Flask,request,render_template,url_for,redirect,session,flash
from werkzeug.utils import secure_filename
import torch
import database
import fileSave
import os
import main
from multiprocessing import Process,Pipe

app = Flask(__name__)
app.secret_key = 'secretkey'
gan = main.maker()
flag = 0
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/',methods = ['GET','POST'])
def loginPage():
    global flag
    if(flag == 1):
        database.gpu_change(0)
        torch.cuda.empty_cache()
        flag = 0
    
    if request.method == "GET":
        return render_template('login.html')
    else:
        userid = request.form["username"]
        password = request.form["password"]
        result = database.login(userid,password)
        if result>0:
            session['user'] = userid
            return redirect(url_for('main_html'))
        else:
            flash("아이디와 비밀번호를 다시 확인해 주세요")
            return render_template("login.html")
        
@app.route('/signup',methods = ['GET','POST'])
def signUpPage():
    global flag
    if(flag == 1):
        database.gpu_change(0)
        torch.cuda.empty_cache()
        flag = 0
    return render_template('signup.html')

#process after signup 
@app.route('/signupResult',methods = ['GET','POST'])
def signupResult():
    global flag
    if(flag == 1):
        database.gpu_change(0)
        torch.cuda.empty_cache()
        flag = 0
    
    if request.method == "POST":
        userid = request.form["username"]
        password = request.form["password"]
        database.insert_data(userid,password)
        fileSave.makeFolder(userid) # folder for gan and images
    return render_template('endsignup.html')

#main page
@app.route('/main',methods = ['GET','POST'])
def main_html():
    global flag
    if(flag == 1):
        database.gpu_change(0)
        torch.cuda.empty_cache()
        flag = 0
    
    if request.method == "POST":
        image_path = './static/images/'
        #argument mode 1: 원하는 헤어스타일 고르기 mode 2: 염색 
        mode = int(request.form.get("type-select"))
        hair_style = int(request.form.get("hair-select"))
        color = [0,0,0]
        if(mode == 1):
            if(hair_style == 0):
                hair_mode = "ref_styling"
            else:
                hair_mode = "latent_styling"
                hair_style = hair_style-1
        elif(mode == 2):
            colorCheck = request.form.get("color-check")
            if(colorCheck == None):
                hair_mode = "ref_dyeing"
            else:
                hair_mode = "RGB_dyeing"
                rgb = request.form.get("rgb-color")
                r = int("0x"+rgb[1:3],16)
                g = int("0x"+rgb[3:5],16)
                b = int("0x"+rgb[5:7],16)
                color = [b,g,r]
        
        name = session['user']
        if(os.path.isfile(image_path+name+'/result.png')):
            os.remove(image_path+name+'/result.png')
        save_path = "./dataset/"+name
        src = request.files['sourceFile']
        src.save(save_path+"/src/src/"+secure_filename('src.jpg'))
        ref = request.files['referenceFile']
        ref.save(save_path+"/ref/ref/"+secure_filename('ref.jpg'))
        #gan start 여기에!!
        parent_proc, child_proc = Pipe()
        heavy_process = Process(target=gan.run(name,hair_mode,child_proc,hair_style,color),daemon=True)
        heavy_process.start()
        flag = parent_proc.recv()

        return redirect(url_for('result_html'))
    else:
        return render_template('main.html')

@app.route('/result')
def result_html():
    global flag
    if flag == 1:
        database.gpu_change(0)
        torch.cuda.empty_cache()
        flag = 0
    username = str(session['user'])
    url = '/static/images/'+username+'/result.png'
    return render_template('result.html',data = url)


if __name__ == '__main__':
    #gan.runTest('test','ref_styling')
    database.gpu_change(0)
    app.run()
