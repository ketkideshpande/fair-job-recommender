import os
import textract
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel

resume_path = '/Users/Ketki/Documents/MS Theses/Reference Papers and Data/Selected/'
jd_input = '/Users/Ketki/Documents/MS Theses/Reference Papers and Data/jd_india1.docx'
reference = '/Users/Ketki/Documents/MS Theses/Candidate list.xlsx'

def preprocessText(text, textEncoding = 'windows-1252'):
    text = re.sub(r'[^\x20-\x7e]',r' ', text.decode(encoding = textEncoding))
    return text


def getResumeReference(referencePath):
    refDataFrame = pd.read_excel(reference)
    return refDataFrame

def getResumeDataFrame(resumePath):
    resume_list = []
    r_file_list = []
    
    for file in os.listdir(resume_path):
        if file == '.DS_Store':
            continue
            
        filename = resume_path + file
        print(filename)
        r_file_list.append(file)
        
        try:
            text = textract.process(filename)
            text = preprocessText(text, 'utf-8')
            resume_list.append(text)
        except Exception as se:
            print('{} -> {}'.format(file, se))
            
    resumeDataFrame = pd.DataFrame(list(zip(r_file_list,resume_list)), \
                             columns = ['res_name','res_contents'])
    refDataFrame = getResumeReference(reference)
    
    resumeDataFrame = resumeDataFrame.join(refDataFrame.set_index('res_name'),\
                                           on = 'res_name')
    
    return resumeDataFrame


def getJobDescription(jobFile):
    jobTitle = 'input'
    jobName = 'JDIndia'
    nationality = 'India'
    jobText = textract.process(jobFile)
    jobText = preprocessText(jobText, 'utf-8')
    return {'res_name': jobTitle, 
            'res_contents': jobText, 
            'name': jobName, 
            'nationality':  nationality}


def TFIDF(processDataFrame):
    
    tfidf = TfidfVectorizer(stop_words = 'english', lowercase = True)
    tfidfMatrix = tfidf.fit_transform(processDataFrame['res_contents'])
    
    cosine_sim = linear_kernel(tfidfMatrix, tfidfMatrix)
    indices = pd.Series(processDataFrame.index, \
                        index=processDataFrame['res_name']).drop_duplicates()
    
    idx = indices['input']
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    resume_indices = [i[0] for i in sim_scores]
    print(processDataFrame['nationality'].iloc[resume_indices])
    print(sim_scores)
    
    return tfidfMatrix

def getKMeans(processDataFrame, tfidfMatrix):
    k = 3
    km = KMeans(n_clusters = k,init = 'random', max_iter = 250, \
                n_init = 15, random_state = 100)
    y_km = km.fit_predict(tfidfMatrix)
    centroids = km.cluster_centers_


    km_result = pd.DataFrame(km.labels_)
    km_validate = pd.concat([processDataFrame['nationality'],km_result], axis = 1) 
    print(km_validate)
    
    return km

def getTSNE(tfidfMatrix):
    k = 30
    svd_matrix = TruncatedSVD(n_components = k, \
                              random_state = 0).fit_transform(tfidfMatrix)


    t_sne = TSNE(perplexity = 12, \
                 verbose = 2, \
                 learning_rate = 200).fit_transform(svd_matrix)
    return t_sne
    
def plot(km, t_sne):
    fig = plt.figure(figsize = (10,10))
    for i in range(0, t_sne.shape[0]):
        if km.labels_[i] == 0:
            c1 = plt.scatter(t_sne[i,0],t_sne[i,1],c='r', marker='+')
        elif km.labels_[i] == 1:
            c2 = plt.scatter(t_sne[i,0],t_sne[i,1],c='g', marker='o')
        elif km.labels_[i] == 2:    
            c3 = plt.scatter(t_sne[i,0],t_sne[i,1],c='b', marker='*')
    plt.grid()
    plt.show()  

def processJobDescription(resumePath, jobFile):
    
    resumeDataFrame = getResumeDataFrame(resumePath)
    jobDescription = getJobDescription(jobFile)
    processDataFrame = resumeDataFrame.append(jobDescription, \
                                              ignore_index = True)
    processDataFrame = processDataFrame.dropna()
    
    
    tfidfMatrix = TFIDF(processDataFrame)
    kMeans = getKMeans(processDataFrame, tfidfMatrix)
    tsne = getTSNE(tfidfMatrix)
    plot(kMeans, tsne)
    
if __name__ == '__main__':
    processJobDescription(resume_path, jd_input)