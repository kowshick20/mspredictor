from flask import Flask, request,render_template,url_for, redirect
import pandas as pd
from ast import literal_eval
from control import *

app = Flask(__name__)
@app.route('/')
@app.route('/student', methods=["GET", "POST"])
def student():
    stud_details= {}
    if request.method == "GET":
        
        universities = get_universities()
        return render_template('student.html', universities=universities)
    elif request.method == "POST":
        
        stud_details["course"] = request.form["course"]
        stud_details["gre_quants"] = request.form["gre_quants"]
        stud_details["gre_verbal"] = request.form["gre_verbal"]
        stud_details["gre"]= int(stud_details["gre_quants"])+int(stud_details["gre_verbal"])
        stud_details["english_mode"] = request.form["english_mode"]
        stud_details["english_score"] = request.form["english_score"]
        stud_details["ugscore"] = request.form["ugscore"]
        stud_details["ugmode"] = request.form["ugmode"]
        stud_details["workex"] = request.form["workex"]
        stud_details["term"] = request.form["term"]
        stud_details["university_choice"] = request.form["university_choice"]
        stud_details["papers_published"] = request.form["papers_published"]
        stud_details["SOP"]= request.form["SOP"]
        stud_details["LOR"]= request.form["LOR"]
        df_student = pd.DataFrame.from_dict([stud_details])
        df_student = runDask(df_student) # For scaling IELTS and TOEFL Score
        
        pred = student_admit_predict(df_student)

        return redirect(url_for("student_predict", stud_details=stud_details, pred=pred))
        print(df_student)

@app.route("/student_predict", methods=["GET", "POST"])
def student_predict():
    if request.method == "GET":
        stud_details = literal_eval(request.args.get('stud_details'))
        df_student = pd.DataFrame.from_dict([stud_details])
        df_with_pred = df_student
        data = df_with_pred.to_dict(orient="records")
        headers = df_with_pred.columns
        return render_template("student_predict.html", data=data,univ_pred="None", headers=headers,
                                pred=request.args.get('pred'))
    elif request.method == "POST":
        stud_details = literal_eval(request.form["data"])
        df_student = pd.DataFrame.from_dict(stud_details)
        df_student = ielts_to_toefl(df_student) # For scaling IELTS and TOEFL Score
        df_student = to_cgpa(df_student) # For percentage to cgpa
        df_with_recommend = student_admit_recommend(df_student)
        univ_pred = df_with_recommend.to_dict(orient="records")
        data = df_student.to_dict(orient="records")
        headers = df_with_recommend.columns
        return render_template("student_predict.html", data=data, univ_pred=univ_pred, headers=headers,
                               username=request.form['username'], pred=request.form['pred'])

if __name__ == "__main__":
    app.run(debug=True)
