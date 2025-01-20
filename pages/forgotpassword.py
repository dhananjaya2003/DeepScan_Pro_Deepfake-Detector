from flet import *
from util.colors import *
from util.validation import *
#from service_a.database_a import *
from service_a.auth_user import *  

class Forgotpass(Container):
    def __init__(self, page: Page):
        super().__init__()
        page.scroll = "adaptive",
        self.page = page  
        self.expand = True
        self.validator = Validator()
        self.bgcolor = "white"
        self.error_border = border.all(width=1, color='red')
        self.alignment = alignment.center
        
        self.email_ref = Ref[TextField]()  

        self.email_box = Container(
            width=400,
            alignment=alignment.center,
            content=TextField(
                ref=self.email_ref,
                border=InputBorder.NONE,
                hint_text="Enter registered email",
                content_padding=padding.only(top=0, bottom=5, right=40, left=30),
                hint_style=TextStyle(size=12, color="#858796"),
                cursor_color='#858796',
                text_style=TextStyle(size=16, color=colors.INDIGO),
            ),
            border=border.all(1, color='black'),
            border_radius=10
        )

        self.content = Column(
            height=770,
            alignment='center',
            horizontal_alignment="center",
            controls=[
                Container(
                    width=800,
                    border_radius=12,
                    padding=40,
                    bgcolor=colors.BLUE_100,
                    content=Column(
                        alignment='center',
                        horizontal_alignment="center",
                        controls=[
                            Text(
                                value="Forgot Your Password?",
                                color='black',
                                size=20,
                                text_align='center',
                                weight=FontWeight.BOLD
                            ),
                            Text(
                                value="Just enter your email address below and we'll send you a link to reset your password!",
                                color='black',
                                size=16,
                                text_align='center'
                            ),
                            self.email_box,
                            Container(
                                alignment=alignment.center,
                                bgcolor="Black",
                                height=40,
                                width=150,
                                border_radius=30,
                                content=Text(
                                    value="Reset Password",
                                    color='white',
                                    weight=FontWeight.BOLD
                                ),
                                on_click=self.reset_password
                            ),
                            Container(height=0),
                            Container(
                                content=Text(
                                    value="Back to Login",
                                    color='blue',
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

    def reset_password(self, e):
        email = self.email_ref.current.value  

        if not self.validator.is_valid_mail(email):
            self.email_box.border = self.error_border
            self.show_error_message("Please enter a valid email address.")
            return

        self.email_box.border = None 

        try:
            auth.send_password_reset_email(email)
            self.show_success_message("Password reset email sent! Check your inbox.")
        except Exception as ex:
            self.show_error_message(f"Failed to send reset email: {str(ex)}")

    def show_error_message(self, message: str):
        self.page.snack_bar = SnackBar(
            Text(message, size=12, color="white"),
            bgcolor="red"
        )
        self.page.snack_bar.open = True
        self.page.update()

    def show_success_message(self, message: str):
        self.page.snack_bar = SnackBar(
            Text(message, size=12, color="white"),
            bgcolor="green"
        )
        self.page.snack_bar.open = True
        self.page.update()
