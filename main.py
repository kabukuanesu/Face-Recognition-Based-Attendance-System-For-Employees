import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import random

# Defining Flask App
app = Flask(__name__)


# Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")


def datetoday2():
    return date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.isdir('Database'):
    os.makedirs('Database')
if f'Attendance-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday()}.csv', 'w') as f:
        f.write('ID,Name,Time_In,Time_Out')
if f'Payments-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Payments-{datetoday()}.csv', 'w') as f:
        f.write('ID,Hours,Wage')
if 'Database.csv' not in os.listdir('Database'):
    with open('Database/Database.csv', 'w') as f:
        f.write('Name,Gender,Role,Cell,Email,Password')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    id = df['ID']
    name = df['Name']
    time_in = df['Time_In']
    time_out = df['Time_Out']
    l = len(df)
    return id, name, time_in, time_out, l


# Extract info from today's payments file in attendance folder
def extract_payments():
    df = pd.read_csv(f'Attendance/Payments-{datetoday()}.csv')
    id = df['ID']
    hours = df['Hours']
    wage = df['Wage']
    l = len(df)
    return id, hours, wage, l


# Extract info from the Database
def extract_login():
    df = pd.read_csv('Database/Database.csv')
    name = df['Name']
    gender = df['Gender']
    role = df['Role']
    cell = df['Cell']
    email = df['Email']
    password = df['Password']
    l = len(df)
    return name, gender, role, cell, email, password, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    time_inside = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    if int(userid) not in list(df['ID']):
        with open(f'Attendance/Attendance-{datetoday()}.csv', 'a') as f:
            f.write(f'\n{userid},{username},{time_inside},{""}')


# Checkout a specific user
def checkout_attendance(name):
    username = name.split('_')[0]
    time_outside = datetime.now().strftime("%H:%M:%S")

    # Reading the csv file and set the index to the "Name" column
    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv', index_col='Name')

    # Updating a cell based on the index (Name) and column
    df.at[username, 'Time_Out'] = time_outside

    # Reset index to 0, 1, 2,...
    df = df.reset_index()

    # Writing the changes into the file
    df.to_csv(f'Attendance/Attendance-{datetoday()}.csv', index=False)


def get_sec(time_str):
    """Get seconds from time"""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def create_next_id():
    unique_id = random.randint(1000, 9999)
    return unique_id


# ROUTING FUNCTIONS

# Our login page
@app.route('/')
def login():
    return render_template('login.html')


# Our main page
@app.route('/home')
def home():
    id, name, time_in, time_out, l = extract_attendance()
    return render_template('home.html', id=id, name=name, time_in=time_in, time_out=time_out, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())


# Our management page
@app.route('/admin')
def admin():
    id, hours, wage, l = extract_payments()
    return render_template('admin.html', id=id, hours=hours, wage=wage, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())


# Our signup page
@app.route('/staff')
def staff():

    return render_template('staff.html')


# This function will run when we click on Check-In Button
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2(),
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        if extract_faces(frame) != ():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    id, name, time_in, time_out, l = extract_attendance()
    return render_template('home.html', id=id, name=name, time_in=time_in, time_out=time_out, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())


# This function will run when we click on Check-Out Button
@app.route('/end', methods=['GET'])
def end():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2(),
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        if extract_faces(frame) != ():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            checkout_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    id, name, time_in, time_out, l = extract_attendance()
    return render_template('home.html', id=id, name=name, time_in=time_in, time_out=time_out, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())


# This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = create_next_id()
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    id, name, time_in, time_out, l = extract_attendance()
    return render_template('home.html', id=id, name=name, time_in=time_in, time_out=time_out, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())


# This function will run when we click on Submit Button
@app.route('/calculate', methods=['GET', 'POST'])
def calculate():
    money = int(request.form['money'])

    id, name, time_in, time_out, l = extract_attendance()
    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv', error_bad_lines=False)
    for i in range(l):
        id = df['ID'][i]
        time_in = df['Time_In'][i]
        time_out = df['Time_Out'][i]
        time_in = get_sec(time_in)
        time_out = get_sec(time_out)
        hours = (time_out - time_in) / 3600
        wage = hours * money
        with open(f'Attendance/Payments-{datetoday()}.csv', 'a') as f:
            f.write(f'\n{id},{hours},{wage}')

    id, hours, wage, l = extract_payments()
    return render_template('admin.html', id=id, hours=hours, wage=wage, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())


# This function will run when we click on login button
@app.route('/logs', methods=['GET', 'POST'])
def logs():
    form_name = request.form['username']
    form_password = request.form['password']

    name, gender, role, cell, email, password, l = extract_login()
    df = pd.read_csv('Database/Database.csv', error_bad_lines=False)
    for i in range(l):
        name = df['Name'][i]
        gender = df['Gender'][i]
        role = df['Role'][i]
        cell = df['Cell'][i]
        email = df['Email'][i]
        password = df['Password'][i]

        if name == form_name and password == form_password:
            id, name, time_in, time_out, l = extract_attendance()
            return render_template('home.html', id=id, name=name, time_in=time_in, time_out=time_out, l=l,
                                   totalreg=totalreg(),
                                   datetoday2=datetoday2())

    return render_template('login.html')


# This function will run when we click on add staff button
@app.route('/sign', methods=['GET', 'POST'])
def sign():
    name = request.form['name']
    gender = request.form['gender']
    role = request.form['role']
    cell = request.form['cell']
    email = request.form['email']
    password = request.form['password']

    with open('Database/Database.csv', 'a') as f:
        f.write(f'\n{name},{gender},{role},{cell},{email},{password}')

    id, name, time_in, time_out, l = extract_attendance()
    return render_template('home.html', id=id, name=name, time_in=time_in, time_out=time_out, l=l,
                           totalreg=totalreg(),
                           datetoday2=datetoday2())


# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
