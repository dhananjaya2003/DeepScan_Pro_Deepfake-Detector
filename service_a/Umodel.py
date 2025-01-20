class UserModel:
    def __init__(self,userid:str,username:str,email:str,password:str):
        self.userid = userid
        self.username = username
        self.email = email
        self.password = password
        
    def toMap(self):
        return {
            'userid':self.userid,
            'username':self.username,
            'email':self.email,
            'password':self.password
        }
    
    @staticmethod
    def fromMap(map:dict):
        return UserModel(
            userid=map['userid'],
            username=map['username'],
            email=map['email'],
            password=map['password']
        )