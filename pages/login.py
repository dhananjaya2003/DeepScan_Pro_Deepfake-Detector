from flet import *
from util.colors import *
from util.validation import *

import re
from service_a.auth_user import *

class Login(Container):
    def __init__(self, page: Page):
        super().__init__()
        self.page = page
        self.is_email_valid = False
        self.is_pass_valid = False
        self.login_email_ref = Ref[TextField]()
        self.login_pass_ref = Ref[TextField]()
        self.validator = Validator()
        self.error_border = border.all(width=1, color='red')
        self.default_border_color = '#bdcbf4'
        self.email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        page.title = 'DeepScan Pro'
        self.expand = True
        Page.padding = 0
        self.page.theme_mode = ThemeMode.DARK
        self.content = Column(
            controls=[
                Container(
                    height =770,
                    image_src= r'assets\bg_DS.png',
                    image_fit=ImageFit.COVER,
                    expand=True,
                    alignment=alignment.center,
                    content=self._build_login_form(),
                )
            ]
        )

    def _build_login_form(self):
        return Column(
            alignment="center",
            horizontal_alignment="center",
            controls=[
                Image(
                    src=r"assets\DS.png",  
                    width=400,
                    height=200,
                    fit=ImageFit.CONTAIN,
                ),
                Container(
                    width=400,
                    padding=20,
                    #bgcolor="white",
                    border_radius=10,
                    blur = Blur(10,12,BlurTileMode.MIRROR),
                    border = border.all(1,'black'),
                    alignment=alignment.top_right,
                    content=Column(
                        horizontal_alignment="center",
                        controls=[
                            Text(
                                value="Welcome Back!",
                                size=20,
                                color="black",
                                text_align="center",
                                weight=FontWeight.BOLD,
                            ),
                            self._build_text_field("Enter Your Email", icons.PERSON, self.login_email_ref),
                            self._build_text_field("Enter Your Password", icons.LOCK, self.login_pass_ref, True,self.login),
                            Container(
                                alignment=alignment.center_right,
                                padding=padding.only(right=20, top=0),
                                content=Text(
                                    value="Forgot Password?",
                                    color="#4e73df",
                                    size=12,
                                ),
                                on_click=lambda _: self.page.go("/forgotpassword"),
                            ),
                            ElevatedButton(
                                    text="Login",
                                    color="white",
                                    bgcolor="#4e73df",
                                    height=40,
                                    width=150,
                                    content=Text(
                                        value="Login", weight=FontWeight.BOLD
                                    ),
                                    on_click=self.login,
                                    style=ButtonStyle(
                                        shape=RoundedRectangleBorder(radius=30),  
                                    )
                                ),

                            Container(height=10),
                            Container(
                                content=Text(
                                    value="Don't have an account? Register",
                                    color="#4e73df",
                                    size=12,
                                    weight=FontWeight.BOLD,
                                ),
                                on_click=lambda _: self.page.go("/signup"),
                            ),
                        ],
                    ),
                ),
                Container(height=10),
                IconButton(
                    icon=icons.ARROW_LEFT_ROUNDED,
                    icon_color=colors.GREY_800,
                    icon_size=30,
                    bgcolor = colors.GREY_300,
                    on_click=lambda _: self.page.go("/startpage"),
                    style=ButtonStyle(
                        CircleBorder(),
                    ),
                ),
                
            ],
        )

    def _build_text_field(self, hint, icon, ref, is_password=False,on_submit_val=None):
        return Container(
            height=40,
            border=border.all(width=1, color=self.default_border_color),
            border_radius=10,
            content=TextField(
                ref=ref,
                border=InputBorder.NONE,
                hint_text=hint,
                hint_style=TextStyle(size=12, color="#858796"),
                cursor_color="#858796",
                prefix_icon=icon,
                text_style=TextStyle(size=16, color="black"),
                password=is_password,
                can_reveal_password=True,
                content_padding=padding.only(top=5, bottom=0, right=40, left=10),
                on_submit=on_submit_val,
            ),
        )


    def login(self, e):
        email = self.login_email_ref.current.value
        password = self.login_pass_ref.current.value

        if not re.fullmatch(self.email_regex, str(email)):
            self.show_error_snackbar("Enter a valid email!")
            return

        if len(password) <= 6:
            self.show_error_snackbar("Password must be more than 6 characters!")
            return

        try:
            self.user = auth.sign_in_with_email_and_password(email, password)
            refreshToken = self.user['refreshToken']
            self.page.client_storage.set('auth-token', refreshToken)
            auth.current_user = self.user

            if self.user:
                self.page.snack_bar = None
                self.page.go('/check')

        except Exception as e:
            #print(e)
            self.show_error_snackbar("Invalid email or Password!")


    def show_error_snackbar(self, message: str):
        self.page.snack_bar = SnackBar(
            content=Text(message, size=12, color="white"),
            bgcolor="red",
            behavior="floating",
            duration=3000
        )
        self.page.snack_bar.open = True
        self.page.update()
