import os  

'''
获取总图片数量
'''
       
def listdir(path, list_name):
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path, list_name)  
        else:  
            list_name.append(file_path) 

l = []
listdir('./3350MMG', l)
print(len(l))