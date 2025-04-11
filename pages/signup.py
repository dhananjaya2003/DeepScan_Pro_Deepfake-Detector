from flet import *
from util.colors import *
from util.validation import *
#from service_a.database_a import create_user, add_user_to_database
import re
from service_a.Umodel import UserModel
from service_a.auth_user import auth,database

class Signup(Container):
    def __init__(self, page: Page):
        super().__init__()
        page.scroll = "adaptive",
        self.is_name = False
        self.is_email = False
        self.is_pass = False
        self.email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        self.fullname_ref = Ref[TextField]()
        self.register_email_ref = Ref[TextField]()
        self.register_pass_ref = Ref[TextField]()
        self.expand = True
        self.alignment = alignment.center
        self.bgcolor = deep_blue
        self.error_border = border.all(width=1, color='red')

        self.name_box = Container(
            width=400,
            height=40,
            content=TextField(
                border=InputBorder.NONE,
                ref=self.fullname_ref,
                content_padding=padding.only(top=5, bottom=0, right=30, left=50),
                hint_text="Enter Your First Name & Last Name",
                hint_style=TextStyle(size=12, color="#858796"),
                cursor_color='#858796',
                text_style=TextStyle(size=16, color='black'),
                prefix_icon=icons.PERSON,
            ),
            border=border.all(1, color='#bdcbf4'),
            border_radius=20,
            alignment=alignment.center
        )

        self.email_box = Container(
            width=400,
            height=40,
            content=TextField(
                border=InputBorder.NONE,
                ref=self.register_email_ref,
                content_padding=padding.only(top=5, bottom=0, right=20, left=20),
                hint_text="Enter Your Email",
                hint_style=TextStyle(size=12, color="#858796"),
                cursor_color='#858796',
                text_style=TextStyle(size=16, color='black'),
                prefix_icon=icons.EMAIL,
            ),
            border=border.all(1, color='#bdcbf4'),
            border_radius=20,
            alignment=alignment.center
        )

        self.pass_box = Container(
            width=400,
            height=40,
            content=TextField(
                ref=self.register_pass_ref,
                border=InputBorder.NONE,
                content_padding=padding.only(top=5, bottom=0, right=20, left=20),
                hint_text="Enter Your Password",
                hint_style=TextStyle(size=12, color="#858796"),
                cursor_color='#858796',
                text_style=TextStyle(size=16, color='black'),
                password=True,
                can_reveal_password=True,
                prefix_icon=icons.LOCK,
            ),
            border=border.all(1, color='#bdcbf4'),
            border_radius=20,
            alignment=alignment.center
        )

        self.content = Column(
            alignment='center',
            horizontal_alignment='center',
            height=770,
            controls=[
                Container(
                    width=500, padding=40, bgcolor='white',
                    border_radius=10,
                    content=Column(
                        horizontal_alignment='center',
                        controls=[
                            Text(
                                value='Create Your Account',
                                size=20,
                                color='black',
                                text_align='center',
                                weight=FontWeight.BOLD
                            ),
                            self.name_box,
                            self.email_box,
                            self.pass_box,
                            Container(
                                alignment=alignment.center,
                                bgcolor="#4e73df",
                                height=40,
                                width=150,
                                border_radius=30,
                                content=Text(
                                    value="Sign Up",
                                    color='white',
                                    weight=FontWeight.BOLD
                                ),
                                on_click=self.signup
                            ),
                            Container(height=20),
                            Container(
                                content=Text(
                                    value="Already have an account? Login",
                                    color='#4e73df',
                                    size=12,
                                    weight=FontWeight.BOLD
                                ),
                                on_click=lambda _: self.page.go('/login')
                            )
                        ]
                    )
                )
            ]
        )

    def signup(self, e):
        name = self.fullname_ref.current.value
        email = self.register_email_ref.current.value
        password = self.register_pass_ref.current.value

        # Validation checks
        if len(name) <= 3:
            self.show_error_snackbar("Enter valid name!")
            self.fullname_ref.current.update()
        else:
            self.is_name = True

        if re.fullmatch(self.email_regex, str(email)):
            self.is_email = True
        else:
            self.show_error_snackbar("Enter valid email!")
            self.register_email_ref.current.update()

        if len(password) <= 6:
            self.show_error_snackbar("Password length must be above 6 chars!")
            self.register_pass_ref.current.update()
        else:
            self.is_pass = True

        try:
            self.user = auth.create_user_with_email_and_password(email, password)
            self.userid = self.user['localId']
            
            if self.user:
                self.userModel = UserModel(
                    userid=self.userid,
                    username=name,
                    email=email,
                    password=password,
                )
                database.child('users').child(self.userid).set(self.userModel.toMap())
                self.show_success_dialog()
        except Exception as e:
            error_message = str(e)
            if "EMAIL_EXISTS" in error_message:
                self.show_error_snackbar("User creation failed. User Already Exists")
            else:
                self.show_error_snackbar(f"User creation failed. Error: {error_message}")

    def show_error_snackbar(self, message):
        self.page.snack_bar = SnackBar(
            content=Container(
                content=Text(message, size=14, color='white', text_align="center"),
                alignment=alignment.center,
                padding=10,
            ),
            bgcolor="red",
            open=True,
            behavior=SnackBarBehavior.FIXED,
            
        )
        self.page.update()

    def show_success_dialog(self):
        dialog = AlertDialog(
            title=Text("Registration Successful"),
            content=Text("You have registered successfully! You can now log in with your credentials."),
            actions=[
                TextButton("OK", on_click=lambda e: self.on_success_dialog_dismiss(dialog))
            ]
        )
        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def on_success_dialog_dismiss(self, dialog):
        dialog.open = False
        self.page.update()
        self.page.go('/login')  # Redirect to the login page
