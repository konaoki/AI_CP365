import numpy as np
import pandas as pd
import Rating
class MovieTester:
    nmovie=10
    print('loading data...')
    movnames=pd.read_csv('data/movie_titles.txt', usecols=range(3),header = None,encoding = "ISO-8859-1")
    movdict_list=[]
    for i in range(nmovie):
        movdat=pd.read_csv('data/mv_'+str(i+1).zfill(7)+'.txt', skiprows=1,header = None,encoding = "utf-8")
        nusr=movdat.shape[0]
        for j in range(nusr):
            movdict=dict()
            movdict['user']=movdat.iloc[j,0] #user id
            movdict['rating']=movdat.iloc[j,1] #rating
            movdict['movie']=movnames.iloc[i,2] #movie name
            movdict_list.append(movdict)
    data=pd.DataFrame(movdict_list)
    print('done.')
    rater = Rating.Rating(25,0.3,5)
    rater.fit(data)
    errors=[]
    movies_to_predict=movnames.iloc[:nmovie,2].values
    for m in range(nmovie):
        watched_userIDs = data.loc[data['movie']==movies_to_predict[m],'user'].values
        for i in range(len(watched_userIDs)):
            errors.append(rater.predict(watched_userIDs[i],movies_to_predict[m])-rater.actual(watched_userIDs[i],movies_to_predict[m]))
        print('-------------------------')
        print("Predicted ratings are off by "+str(np.average(errors))+"+/-"+str(np.std(errors)/len(watched_userIDs)**0.5)+" on average for "+movies_to_predict[m]+".")
        print("People who like "+movies_to_predict[m]+" might also like "+rater.recommend(movies_to_predict[m])+".")
