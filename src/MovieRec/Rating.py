import pandas as pd
import numpy as np
class Rating:
    def __init__(self,d,lr,i):
        self.d=d
        self.lr=lr
        self.iteration=i
    def fit(self,df): #df: user,rating,movie
        d=self.d
        lr=self.lr
        movieIDs = df['movie'].unique() #all unique movie ids
        userIDs=df['user'].unique() #all unique user ids
        movie_lookup=dict(zip(movieIDs,np.arange(0,len(movieIDs))))
        user_lookup=dict(zip(userIDs,np.arange(0,len(userIDs))))
        movie_matrix = np.empty((len(movieIDs),d)) #width d, height len(movieIDs)
        user_matrix = np.empty((len(userIDs),d)) #width d, height len(userIDs)
        user_matrix.fill(0.1)
        movie_matrix.fill(0.1)
        avr_ratings=dict()
        for epoch in range(self.iteration):
            for idx,id in enumerate(movieIDs):
                watched_userIDs = df.loc[df['movie']==id,'user'] #extract all users who watched this movie from df
                watched_user_matrix = user_matrix[[user_lookup[x] for x in watched_userIDs],:]
                ratings = df.loc[df['movie']==id,'rating']
                avr = np.average(ratings)
                avr_ratings[id]=avr
                utm=np.dot(watched_user_matrix, np.reshape(movie_matrix[idx],(d,1)))
                rating_diff=np.subtract(avr*np.ones(len(watched_userIDs)),ratings)
                utm_plus_diff=np.add(np.reshape(utm,len(watched_userIDs)), rating_diff)
                sum=np.dot(np.transpose(watched_user_matrix),utm_plus_diff)
                #print(pd.DataFrame(watched_user_matrix))
                #print(pd.DataFrame(utm_plus_diff))
                movie_matrix[idx]=movie_matrix[idx]-2*lr/len(watched_userIDs)*sum
            for idx,id in enumerate(userIDs):
                watched_movieIDs = df.loc[df['user']==id,'movie'] #extract all movies this user watched from df
                watched_movie_matrix = movie_matrix[[movie_lookup[x] for x in watched_movieIDs],:]
                ratings = df.loc[df['user']==id,'rating']
                avrs = [avr_ratings[x] for x in watched_movieIDs]
                utm=np.dot(watched_movie_matrix, np.reshape(user_matrix[idx],(d,1)))
                rating_diff=np.subtract(avrs,ratings)
                utm_plus_diff=np.add(np.reshape(utm,len(watched_movieIDs)), rating_diff)
                sum=np.dot(np.transpose(watched_movie_matrix),utm_plus_diff)
                #print(pd.DataFrame(watched_user_matrix))
                #print(pd.DataFrame(utm_plus_diff))
                user_matrix[idx]=user_matrix[idx]-2*lr/len(watched_movieIDs)*sum
            self.df=df
            self.movie_lookup=movie_lookup
            self.user_lookup=user_lookup
            self.movieIDs=movieIDs
            self.movie_matrix=movie_matrix
            self.user_matrix=user_matrix
            print("epoch "+str(epoch+1)+" finished.")


        #print(pd.DataFrame(movie_matrix))
        #print(pd.DataFrame(user_matrix))
    def predict(self,userID,movieID):
        df=self.df
        ave = np.average(df.loc[df['movie']==movieID,'rating'])
        watched_movieIDs = df.loc[df['user']==userID,'movie'] #extract all movies this user watched from df
        watched_movie_matrix = self.movie_matrix[[self.movie_lookup[x] for x in watched_movieIDs],:]
        utm=np.dot(watched_movie_matrix, np.reshape(self.user_matrix[self.user_lookup[userID]],(self.d,1)))
        return np.add(np.reshape(utm,len(watched_movieIDs)),ave)[0]
    def actual(self,userID,movieID):
        df=self.df
        rating=df.loc[(df['movie']==movieID)&(df['user']==userID),'rating'].values.item()
        return rating

    def recommend(self,movieID):
        mids = self.movieIDs
        mvs=[]
        for mid in mids:
            if mid!=movieID:
                self_idx=self.movie_lookup[movieID]
                neighbor_idx=self.movie_lookup[mid]
                v1=self.movie_matrix[self_idx]
                v2=self.movie_matrix[neighbor_idx]
                cos=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                mvs.append([cos,mid])
        #print(mvs)
        maxidx=0
        for i in range(len(mvs)):
            if mvs[i]>mvs[maxidx]:
                maxidx=i
        return mvs[maxidx][1]
