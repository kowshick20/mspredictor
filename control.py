import pandas as pd

import pickle
import numpy as np
import dask

# Method to get list of the Universities
def get_universities():
    univ = pd.read_csv(r"data/university_list.csv")
    univ_dict = univ.to_dict(orient="records")
    return univ_dict

# This method is used to convert percentage and CGPA out of 10 to CGPA out of 4
def to_cgpa(ielts_to_toefl):
    student_df = ielts_to_toefl
    score =student_df.iloc[0]["ugscore"]
    mode = student_df.iloc[0]["ugmode"]
    s = 0
    try:
       score = float(score)
    except:
       score= 0
    if mode == 'ugscore_perc':
       s = ((score)/20) - 1
       s = round(s,2)
       student_df["undergraduation_score"]= s
    else:
       s = ((score)/10)*4
       s = round(s,2)
       student_df["undergraduation_score"] = s
    return student_df

# This method is used to scale the ielts score to toefl equivalent
def ielts_to_toefl(df_student):
    if df_student.iloc[0]['english_mode'] == 'ielts' :
        score  = float(df_student.english_score)
        if score <= 9.0:
            eng_score = np.array([score],dtype=float)
            df_student['test_score_toefl'] =pd.cut(eng_score, bins=[-1,0.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], labels=[0,31,34,45,59,78,93,101,109,114,117,120])
        else:
            df_student['test_score_toefl'] = df_student.english_score
    else:
        df_student['test_score_toefl'] = df_student.english_score
    return df_student
'''
Task Creation
'''
def runDask(df_student):
    dsk={'Task0':(ielts_to_toefl,df_student),
         'Task1':(to_cgpa,'Task0')}
    stud_df = dask.multiprocessing.get(dsk, 'Task1')
    return stud_df        

# Method to get the ranking of the university
def get_univ_ranking(univ_name):
    univ_df = pd.read_csv(r'data/ranking.csv')
    univ_dict ={}
    for index, row in univ_df.iterrows():
        univ_dict[row[0]] = univ_dict.get(row[0], row[1])
    univ_rank = univ_dict.get(univ_name[0])
    return univ_rank, univ_dict
    
# Method used by student to predict chance of getting admit from University based on university choice and model
def student_admit_predict(df_student):
    model_path = r'model/student_university_bagging_classifier_predict.pickel'
    with open(model_path, 'rb') as handle:
        model = pickle.load(handle)

        scalar_path = r'data/standard_scaler_bagging_model.pickel'
    with open(scalar_path, 'rb') as handle:
        scalar = pickle.load(handle)
   
    #         # creating the dataframe to predict the result
    column_list = ["gre_score","gre_score_quant","gre_score_verbal","test_score_toefl","undergraduation_score","work_ex","papers_published","ranking"]
    student_df_to_predict = pd.DataFrame(columns=column_list)
    student_df_to_predict['gre_score'] = df_student['gre']
    student_df_to_predict['gre_score_quant'] = df_student['gre_quants']
    student_df_to_predict['gre_score_verbal'] = df_student['gre_verbal']
    student_df_to_predict['test_score_toefl'] = df_student['test_score_toefl'].astype(float)
    student_df_to_predict['undergraduation_score'] = df_student['undergraduation_score']
    student_df_to_predict['work_ex'] = df_student['workex']
    student_df_to_predict['papers_published'] = df_student['papers_published']
    student_df_to_predict['SOP'] = df_student['SOP']
    student_df_to_predict['LOR'] = df_student['LOR']
    univ_rank, univ = get_univ_ranking(df_student['university_choice'])
    student_df_to_predict['ranking'] = univ_rank
    student_df_to_predict= student_df_to_predict.apply(pd.to_numeric)
    
    predictions = model.predict(scalar.transform(student_df_to_predict))
    return predictions[0]
   

#Method used by student to know the list of recommended univeristy
def student_admit_recommend(df_student):
    model_path = r'model/student_university_bagging_classifier_predict.pickel'
    with open(model_path, 'rb') as handle:
        model = pickle.load(handle)

        scalar_path = r'data/standard_scaler_bagging_model.pickel'
    with open(scalar_path, 'rb') as handle:
        scalar = pickle.load(handle)

    column_list = ["gre_score", "gre_score_quant", "gre_score_verbal", "test_score_toefl", "undergraduation_score",
                   "work_ex", "papers_published", "ranking"]
    student_df_to_append = pd.DataFrame(columns=column_list)
    student_df_to_append['gre_score'] = df_student['gre']
    student_df_to_append['gre_score_quant'] = df_student['gre_quants']
    student_df_to_append['gre_score_verbal'] = df_student['gre_verbal']
    student_df_to_append['test_score_toefl'] = df_student['test_score_toefl'].astype(float)
    student_df_to_append['undergraduation_score'] = df_student['undergraduation_score']
    student_df_to_append['work_ex'] = df_student['workex']
    student_df_to_append['papers_published'] = df_student['papers_published']
    student_df_to_append['SOP'] = df_student['SOP']
    student_df_to_append['LOR'] = df_student['LOR']
    univ_rank, univ = get_univ_ranking(df_student['university_choice'])
    student_df_to_recommend = pd.DataFrame(columns=column_list)
    for i in range(0,29):
        student_df_to_recommend = student_df_to_recommend.append(student_df_to_append)
    student_df_to_recommend['ranking'] = list(univ.values())

    prediction = pd.DataFrame(model.predict_proba(scalar.transform(student_df_to_recommend)),
                 columns=['accept', 'reject'])
    prediction['Universities'] = list(univ.keys())
    prediction['Ranking'] = list(univ.values())
    prediction['accept']= (prediction['accept']*100).astype(int)
    prediction = prediction.loc[(prediction['accept'])>55,:]
    best_pred = prediction.sort_values(by=['Ranking'], ascending=True).head(6)
    best_pred = best_pred.sort_values(by=['accept'], ascending=False)
    return best_pred
