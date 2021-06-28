import os

def makeFolder(username):
    data_path = './dataset/'+username
    result_path = './result/'+username
    try:
        os.mkdir(data_path)
        os.mkdir(result_path)
        #dataset
        os.mkdir(data_path+'/src')
        os.mkdir(data_path+'/src/src')
        os.mkdir(data_path+'/ref')
        os.mkdir(data_path+'/ref/ref')
        os.mkdir('./static/images/'+username)
        #label
        folders = ['RGB_dyeing','ref_dyeing','ref_styling','latent_styling']
        for folder in folders:
            os.makedirs(result_path+'/label/'+folder+'/src')
            os.makedirs(result_path+'/label/'+folder+'/ref')
            os.mkdir(result_path+'/{}'.format(folder))
        #result
        os.makedirs(result_path+'/result')
        #styling_ref
        
    except:
        print("already Sign Up ID")
