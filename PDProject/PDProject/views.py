from django.shortcuts import render
import requests
import sys
from subprocess import run,PIPE
from django.core.files.storage import FileSystemStorage
from pyrebase import pyrebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore
from datetime import datetime

cred = credentials.Certificate('//home//subhranil//Pictures//PDProject//demoproject-92352-firebase-adminsdk-y3exc-3beeea35c3.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'demoproject-92352.appspot.com'
})
db = firestore.client()
bucket = storage.bucket()

def button(request):
    return render(request,'index1.html')

def external(request):
    audio = request.FILES['audio']
    fs = FileSystemStorage()
    filename = fs.save(audio.name,audio)  
    fileurl = fs.open(filename)      
    out = run([sys.executable,'//home//subhranil//Pictures//PDProject//actual_project.py',str(fileurl)],shell=False,stdout=PIPE)
    f = open("Results/results.txt","w+")
    f.truncate(0)
    f.write(str(out.stdout.decode('utf-8')))
    f.close()
    blob = bucket.blob('results.txt')
    outfile='Results/results.txt'
    blob.upload_from_filename(outfile)
    return render(request,'index1.html',{'data':out.stdout.decode('utf-8')})