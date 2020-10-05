import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    # apply regex
    def get_title(self, name):
        pattern = ' ([A-Za-z]+)\.'
        title_search = re.search(pattern, name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    def load_data(self):
        # For beginning, transform train['FullDescription'] to lowercase using text.lower()
        self.dataset['title']=self.dataset['title'].str.lower()

        # Then replace everything except the letters and numbers in the spaces.
        # it will facilitate the further division of the text into words.
        self.dataset['title']=self.dataset['title'].replace('[^a-zA-Z0-9]', ' ', regex = True)

        # columns combination
        #self.dataset['FamilySize'] = self.dataset['SibSp'] + self.dataset['Parch'] + 1

        # Adding views_log feature
        #self.dataset['views_log'] = np.log(self.dataset['views'] + 1)

        # Numerical features
        numerical_features=['publish_hour','category_id','likes','dislikes','comment_count',
                    'comments_disabled','ratings_disabled','tag_appeared_in_title_count',
                   'tag_appeared_in_title', 'trend_day_count','trend.publish.diff','trend_tag_highest',
                   'trend_tag_total','tags_count','subscriber','views']

        USvideos_num=self.dataset[numerical_features]

        USvideos_num['comments_disabled']=USvideos_num['comments_disabled'].astype(int)
        USvideos_num['ratings_disabled']=USvideos_num['ratings_disabled'].astype(int)
        USvideos_num['tag_appeared_in_title']=USvideos_num['tag_appeared_in_title'].astype(int)

        # Convert a collection of raw documents to a matrix of TF-IDF features with TfidfVectorizer
        vectorizer = TfidfVectorizer(min_df=1)
        X_tfidf = vectorizer.fit_transform(self.dataset['title'])

        #features = numerical_features

        USvideos_num['likesdislikes_rate']=self.dataset['likes']/(self.dataset['dislikes']+1)
        USvideos_num['likescomments_rate']=self.dataset['likes']/(self.dataset['comment_count']+1)
        USvideos_num['dislikescomments_rate']=self.dataset['dislikes']/(self.dataset['comment_count']+1)
        USvideos_num['attention']=self.dataset['likes']+self.dataset['dislikes']

        USvideos_num['subscriber'].fillna((USvideos_num['subscriber'].mean()), inplace=True)
        USvideos_num['dislikescomments_rate'].fillna((USvideos_num['dislikescomments_rate'].mean()), inplace=True)
        USvideos_num['likescomments_rate'].fillna((USvideos_num['likescomments_rate'].mean()), inplace=True)
        USvideos_num['likesdislikes_rate'].fillna((USvideos_num['likesdislikes_rate'].mean()), inplace=True)

        USvideos_num['subscriber'].fillna((USvideos_num['subscriber'].mean()), inplace=True)

        USvideos_num=USvideos_num.drop(['views'], axis=1)

        features = USvideos_num.columns.to_list()

        tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names())

        X = USvideos_num[features]
        Xnew = pd.concat([X, tfidf], axis=1)


        #features.remove('views')
        #features.remove('views_log')

        # binning with qcut
        #self.dataset['Fare'] = pd.qcut(self.dataset['Fare'], 4)

        # binning with cut
        #self.dataset['Age'] = pd.cut(self.dataset['Age'], 5)

        # apply regex
        #self.dataset['Title'] = self.dataset['Name'].apply(self.get_title)

        # replace
        #self.dataset['Title'] = self.dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
        # replace
        #self.dataset['Title'] = self.dataset['Title'].replace('Mme', 'Mrs')
        # fill nans
        #self.dataset['Title'] = self.dataset['Title'].fillna(0)


        # encode labels
        #le = LabelEncoder()

        #le.fit(self.dataset['Gender'])
        #self.dataset['Gender'] = le.transform(self.dataset['Gender'])

        #self.dataset = USvideos_num


        return Xnew
