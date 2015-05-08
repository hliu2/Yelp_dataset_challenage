import json

#  Business Data
file_business = open('yelp_academic_dataset_business.json')
data_business = [json.loads(line) for line in file_business]
print "total number of business: ",len(data_business)
print json.dumps(data_business[10000],indent=4)

#  User Data
file_user = open('yelp_academic_dataset_user.json')
data_user = [json.loads(line) for line in file_user]
print "total number of users: ",len(data_user)
print json.dumps(data_user[20000],indent=4)

#  Review Data
import json
file_review = open('yelp_academic_dataset_review.json')
data_review = [json.loads(line) for line in file_review]
print "total number of reviews: ",len(data_review)
print json.dumps(data_review[20000],indent=4)

# I am interested in how peopel comment on hotels,so select all the hotel data into data_hotle
# and then write it into txt file
data_hotel=[i for i in data_business if u'Hotels' in i['categories']]
print len(data_hotel)
f=open('data_hotel.txt','w')
data=''
for i in data_hotel:
    data=data+str(i)+'\n'
f.write(data)
f.close()

# plot a histogram of number of stars for the business set
star=[i['stars'] for i in data_hotel]
import matplotlib.pyplot as plt
import pandas as pd
star=pd.DataFrame(star)
star.hist(bins=20)

# Collect all the reviews for all hotel entities, and wirtie it into json file  
business_data_id=[i[u'business_id'] for i in data_hotel]
hotel_review_data=[i for i in data_review if i[u'business_id'] in business_data_id]
len(hotel_review_data)
with open('data_hotels_review.json','w') as outfile:
    json.dumps(hotel_review_data,outfile)
    
# Plot a histogram of different number of stars for the review set.
import pandas as pd
hotel_review_data=pd.read_json('data_hotels_review.json')
star=hotel_review_data.stars
star.hist(bins=20)

# Collect all the text data in these reviews, Use the TfidVectorizer class on 
# the text data above,and convert each review into a feature vector.
text=hotel_review_data.text
from sklearn.feature_extraction.text import TfidfVectorizer
feature=TfidfVectorizer().fit_transform(text)

# Plot a table of the most frequent words for all the reviews with five, four,
# three, two, one stars.
from sklearn.feature_extraction.text import CountVectorizer
five_star_data=text[hotel_review_data.stars==5]
four_star_data=text[hotel_review_data.stars==4]
three_star_data=text[hotel_review_data.stars==3]
two_star_data=text[hotel_review_data.stars==2]
one_star_data=text[hotel_review_data.stars==1]

def sortkey(s):
    return s[1]

five_star_fit=CountVectorizer(min_df=0.1,max_df=0.2).fit(five_star_data)
five_star_feature_count=five_star_fit.transform(five_star_data).toarray().sum(axis=0)
five_star_feature_names=five_star_fit.get_feature_names()
feature_count=zip(five_star_feature_names,five_star_feature_count)
pd.DataFrame(sorted(feature_count,key=sortkey,reverse=True),columns=['word','occurence'],
             index=np.arange(len(feature_count))+1)[:10]

four_star_fit=CountVectorizer(min_df=0.1,max_df=0.2).fit(four_star_data)
four_star_feature_count=four_star_fit.transform(four_star_data).toarray().sum(axis=0)
four_star_feature_names=four_star_fit.get_feature_names()
feature_count=zip(four_star_feature_names,four_star_feature_count)
pd.DataFrame(sorted(feature_count,key=sortkey,reverse=True),columns=['word','occurence'],
             index=np.arange(len(feature_count))+1)[:10]

three_star_fit=CountVectorizer(min_df=0.1,max_df=0.2).fit(three_star_data)
three_star_feature_count=three_star_fit.transform(three_star_data).toarray().sum(axis=0)
three_star_feature_names=three_star_fit.get_feature_names()
feature_count=zip(three_star_feature_names,three_star_feature_count)
pd.DataFrame(sorted(feature_count,key=sortkey,reverse=True),columns=['word','occurence'],
             index=np.arange(len(feature_count))+1)[:10]

two_star_fit=CountVectorizer(min_df=0.1,max_df=0.2).fit(two_star_data)
two_star_feature_count=two_star_fit.transform(two_star_data).toarray().sum(axis=0)
two_star_feature_names=two_star_fit.get_feature_names()
feature_count=zip(two_star_feature_names,two_star_feature_count)
pd.DataFrame(sorted(feature_count,key=sortkey,reverse=True),columns=['word','occurence'],
             index=np.arange(len(feature_count))+1)[:10]

one_star_fit=CountVectorizer(min_df=0.1,max_df=0.2).fit(one_star_data)
one_star_feature_count=one_star_fit.transform(one_star_data).toarray().sum(axis=0)
one_star_feature_names=one_star_fit.get_feature_names()
feature_count=zip(one_star_feature_names,one_star_feature_count)
pd.DataFrame(sorted(feature_count,key=sortkey,reverse=True),columns=['word','occurence'],index
    =np.arange(len(feature_count))+1)[:10]

hotel_reviews = open('data_hotels_review.json')
reviews = [json.loads(line) for line in hotel_reviews]

#Store the text data of all the collected reviews into a local txt file, where each line
#of the file contains the text of one review.
with open('reviews.txt', 'w') as outfile:
    for review in reviews[0]:
        review = review['text'].replace('\n', '')
        review = review.replace(",", "")
        review = review.replace(".", "")
        review = review.replace("!", "")
        review = review.replace("(", "")
        review = review.replace(")", "")
        review = review.replace("?", "")
        review = review.replace(":", "")
        outfile.write(review.encode('UTF-8')+'\n')
        
# find all hotel reviews's users, and store the user data into json file 
user_data_hotel=pd.merge(left=hotel_review_data,right=pd.DataFrame(data_user),how='left',on='user_id')
len(user_data_hotel)
user_data_hotel_json=user_data_hotel.to_json(path_or_buf='user_data_hotel.json')


#Convert the friendship information among the users into a graph, where each node is
# a user, each edge represents the friendship relationship between the two users.
from pygraph.classes.graph import graph
from pygraph.classes.digraph import digraph
from pygraph.algorithms.searching import breadth_first_search
from pygraph.readwrite.dot import write

gr = digraph()

# Add nodes and edges
userid=[]
for i in range(len(data_user)):
    userid.append(data_user[i]['user_id'])

gr.add_nodes(userid)#get all the different user_id

for i in range(len(userid)):
    for j in range(len(data_user[i]['friends'])):
        gr.add_edge((userid[i],data_user[i]['friends'][j]))
        
# apply pagerank algorithm 
def pagerank(graph, damping_factor=0.85, max_iterations=100, min_delta=0.00001):
    """
    Compute and return the PageRank in an directed graph.    
    
    @type  graph: digraph
    @param graph: Digraph.
    
    @type  damping_factor: number
    @param damping_factor: PageRank dumping factor.
    
    @type  max_iterations: number 
    @param max_iterations: Maximum number of iterations.
    
    @type  min_delta: number
    @param min_delta: Smallest variation required to have a new iteration.
    
    @rtype:  Dict
    @return: Dict containing all the nodes PageRank.
    """
    
    nodes = graph.nodes()
    graph_size = len(nodes)
    if graph_size == 0:
        return {}
    min_value = (1.0-damping_factor)/graph_size #value for nodes without inbound links
    
    # itialize the page rank dict with 1/N for all nodes
    pagerank = dict.fromkeys(nodes, 1.0/graph_size)
        
    for i in range(max_iterations):
        diff = 0 #total difference compared to last iteraction
        # computes each node PageRank based on inbound links
        for node in nodes:
            rank = min_value
            for referring_page in graph.incidents(node):
                rank += damping_factor * pagerank[referring_page] / len(graph.neighbors(referring_page))
                
            diff += abs(pagerank[node] - rank)
            pagerank[node] = rank
        
        #stop if PageRank has converged
        if diff < min_delta:
            break
    
    return pagerank

# compute the page rank for each users
Pagerank=pagerank(gr)
Pagerank

# Plot a table of the top 30 users with the largest PageRank scores and their PageRank scores
sort=zip(Pagerank.keys(),Pagerank.values())
pd.DataFrame(sorted(sort,key=sortkey,reverse=True),index=np.arange(len(Pagerank))+1,
             columns=['user_id','pagerank'])[:30]
             
# re-compute the avearage stars of each business through re-weighting each review 
# by users's PageRank scores.
review_stars_data=[{'user_id':review['user_id'],'business_id':review['business_id'],
                    'stars':review['stars']} for review in data_review]

# build a dataframe to store pagerank data and a dataframe to sotre review information 
review_stars=pd.DataFrame(review_stars_data)
user_pagerank_score=Pagerank.values()
user_pagerank_id=Pagerank.keys()

#join these two data frame on user_id 
pagerank_data=pd.DataFrame({'user_pagerank_score':user_pagerank_score,'user_id':user_pagerank_id})
data=pd.merge(pagerank_data,review_stars,on='user_id')

data['weighted']=data.stars*data.user_pagerank_score 
data_sum=data.groupby('business_id')['weighted'].sum()
data_weight_sum=data.groupby('business_id')['user_pagerank_score'].sum()
average_star=data_sum/data_weight_sum
print average_star

# Computer the difference between the new average_star (weighted by PageRank) and 
# the original average_star of each business, Plot a table of the top 20 business entities
# with the largest differences in average_star
original_stars=[business['stars']for business in data_business]
business_id=[business['business_id']for business in data_business]
original_star=pd.Series(original_stars,index=business_id)
sort=zip(difference.index,np.absolute(difference).values)
pd.DataFrame(sorted(sort,key=sortkey,reverse=True),index=np.arange(len(difference))+1,
             columns=['business_id','abs_difference'])[:20]

