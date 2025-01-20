import pyrebase

firebaseConfig = {
  'apiKey': "AIzaSyDaw-vQ3g77HZgDCoe5oWK-LdJqcv18aug",
  'authDomain': "deepscanpro-auth.firebaseapp.com",
  'databaseURL': "https://deepscanpro-auth-default-rtdb.asia-southeast1.firebasedatabase.app",
  'projectId': "deepscanpro-auth",
  'storageBucket': "deepscanpro-auth.firebasestorage.app",
  'messagingSenderId': "261688864421",
  'appId': "1:261688864421:web:319642d4f539b25766e74c",
  'measurementId': "G-YMD2PVW18Y"
};

databaseFirebase = pyrebase.initialize_app(firebaseConfig)
auth = databaseFirebase.auth()
database = databaseFirebase.database()

