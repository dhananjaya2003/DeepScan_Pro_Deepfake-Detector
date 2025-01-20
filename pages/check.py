from flet import *
import webbrowser
import os
from math import pi
from pages.login import *
from service_a.auth_user import *
from trained_models.mymodel import *
from time import sleep
from trained_models.audiomodel import *
from trained_models.videomodel import *

class HomePage(Container):
    def __init__(self, page: Page):
        super().__init__()
        self.page = page
        page.title = "Home"
        page.scroll = None
        page.theme_mode = ThemeMode.LIGHT
        self.t1 = Text("Detecting deepfake....⌛",size=10,visible=False)
        self.progress_detect = ProgressBar(
                    width=250,
                    height=8,
                    color='#6c757d',
                    visible=False, 
                    bgcolor='#eeeeee',
                    border_radius=4,
                    )
        self.selected_file_path = None
        self.uploaded_file_path_text = Text( 
                                            value="",
                                            size=16,
                                            color=colors.BLUE_300,
                                            weight=FontWeight.BOLD,
                                            text_align="center",
                                        )
        self.file_result = Text( 
                                value="",
                                size=16,
                                color=colors.BLUE_300,
                                weight=FontWeight.BOLD,
                                text_align="center",
                                visible=False
                            )
        
        self.red_box = Container(
            width=48,
            height=48,
            bgcolor="red",
            border=border.all(2.5, "black"),
            border_radius=4,
            rotate=transform.Rotate(180, alignment.center),
            #animate_rotation=animation.Animation(700, "easeInOut"),
            
        )

        self.blue_box = Container(
            width=48,
            height=48,
            bgcolor="blue",
            border=border.all(2.5, "black"),
            border_radius=4,
            rotate=transform.Rotate(180, alignment.center),
            #animate_rotation=animation.Animation(700, "easeInOut"),
        )

        self.container_side = Column(
            controls=[
                self.blue_box,
                self.red_box,  
            ],
            alignment="center",
        )
        self.expand = True
        self.bgcolor = '#F9F7F7'
        self.allowed_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".mp3", ".wav", ".mp4", ".avi", ".mkv", ".mov"]
        self.uploaded_file_path_text = Text("")
        self.content = Column(
            height=770,
            scroll= None,
            expand=True,
            controls=[
                self._build_header(),
                Divider(height=1, color=colors.BLACK26),
                Row(
                    expand=True,
                    controls=[
                        self._build_sidebar(),
                        self._build_main_content(),
                    ],
                ),
            ],
        )

    def _build_header(self):
        return Container(
            height=100,
            padding=padding.only(left=50, right=70),
            bgcolor=None,
            content=Row(
                alignment=MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=CrossAxisAlignment.CENTER,
                controls=[
                    Row(
                        controls=[
                            Image(
                                src=r"assets/DS.png",
                                width=200,
                                height=100,
                                fit=ImageFit.CONTAIN,
                            ),
                        ]
                    ),
                    
                    Row(width=650),

                    Row(
                        controls=[
                            Text(
                                value="Home",
                                color="#4e73df",
                                size=20,
                                weight=FontWeight.BOLD,
                            ),

                            PopupMenuButton(
                                content =Text(
                                    value = 'Resources',
                                    color = colors.BLACK,
                                    size = 20,
                                    weight=FontWeight.BOLD,
                                ),
                                items=[
                                    PopupMenuItem(text="Articles",on_click=self._article_link),
                                    PopupMenuItem(text="Blogs",on_click=self._blog_link),
                                    PopupMenuItem(text="GitHub",on_click = self._git_link),
                                    PopupMenuItem(text="Data Source"),

                                ],
                            ),

                            PopupMenuButton(
                                content =Text(
                                    value = 'Use Cases',
                                    color = colors.BLACK,
                                    size = 20,
                                    weight=FontWeight.BOLD,
                                ),
                            ),

                            PopupMenuButton(
                                content =Text(
                                    value = 'About',
                                    color = colors.BLACK,
                                    size = 20,
                                    weight=FontWeight.BOLD,
                                ),
                                items=[
                                    PopupMenuItem(text="Developers",on_click=lambda _: self.page.go("/info"),),
                                    PopupMenuItem(text="Data Privacy",on_click=lambda _: self.page.go('/faq')),
                                ],
                            ),

                            
                        ],
                        spacing=30,

                    ),


                    PopupMenuButton(
                        content=Container(
                            width=50,  # Adjust width and height to make the icon larger
                            height=50,
                            content=Icon(
                                icons.ACCOUNT_CIRCLE,
                                size=40,  # Adjust the size to your preference
                                color=colors.INDIGO_500,
                            ),
                        ),
                        items=[
                            PopupMenuItem(text="User", on_click=self._show_user_info),
                            PopupMenuItem(text="Logout", on_click=self._logout),
                        ],
                    ),
                ],
            ),
        )

    def _build_sidebar(self):
        return Container(
            width=170,
            height=640,
            padding=padding.only(left=20, right=10,top=20),
            alignment=alignment.top_left,
            bgcolor="#261a3a",
            border_radius=10,
            content=Column(
                controls=[
                    Container(
                        content = Image(
                            src= r"C:\Users\hp\Downloads\animation.gif",
                            width=100,
                            height=100,
                            fit=ImageFit.CONTAIN,
                        ),
                        alignment=alignment.center,
                        padding=padding.only(top=10, left=10, right=20), 
                    ),
                    Container(height=40),
                    self._build_sidebar_item(icon=icons.SEARCH, label="Search", on_click=self._open_google),
                    self._build_sidebar_item(icon=icons.LIVE_HELP, label="Help", on_click=self._download_user_manual),
                    self._build_sidebar_item(icon=icons.QUESTION_MARK, label="FAQ's",on_click=lambda _: self.page.go('/faq')),
                    self._build_sidebar_item(icon=icons.BUG_REPORT, label="Report Bug", centered=True, on_click=self._send_bug_report),
                    self._build_sidebar_item(icon=icons.POLICY_OUTLINED, label="Policy",on_click=lambda _: self.page.go('/faq')),
                    self._build_sidebar_item(icon=icons.CONTACT_PAGE_OUTLINED, label="Contact",on_click=lambda _: self.page.go("/info"),),
                    Divider(color=colors.WHITE24),
                    Container(height=60),
                    self._build_sidebar_item(icon=icons.SETTINGS, label="Settings", on_click=self._open_settings_dialog),
                    Divider(color=colors.WHITE24),
                ],
                spacing=15,
                alignment=MainAxisAlignment.START,
                expand=True,
            ),
        )
    
    def _build_sidebar_item(self, icon, label, on_click, centered=False):
        return Container(
            alignment=alignment.center if centered else alignment.center_left,
            content=Row(
                controls=[Icon(icon, color=colors.WHITE, size=20), Text(label, color=colors.WHITE, size=16, weight="bold")],
                spacing=10,
                alignment=MainAxisAlignment.START,
            ),
            on_click=on_click,
        )

    def _build_main_content(self):
        return Container(
            expand=True,
            bgcolor=None,
            padding=padding.only(top=70, left=15, bottom=50),
            content=Row(
                alignment=MainAxisAlignment.START,
                vertical_alignment=CrossAxisAlignment.START,
                controls=[
                    Container(
                        width=700,
                        padding=padding.only(top=50, left=15, right=20),
                        content=Column(
                            controls=[
                                Text(
                                    "\t\t\t\tBe Protected Against Deepfakes!",
                                    size=37,
                                    weight="bold",
                                    color="#000000",
                                ),
                                Text(
                                    "We offer a tool that can identify if an image, video, and audio is a deepfake or real with greater accuracy.",
                                    size=16,
                                    color="#555555",
                                    text_align="center",
                                ),
                                Container(height=10),
                                Row(
                                    alignment=MainAxisAlignment.CENTER,
                                    spacing=50,
                                    controls=[
                                        Container(
                                            bgcolor="#dee2e6",
                                            border_radius=10,
                                            border=border.all(1, 'black12'),
                                            width=170,
                                            height=300,
                                            padding=padding.all(10),
                                            content=Column(
                                                alignment=MainAxisAlignment.START,
                                                horizontal_alignment=CrossAxisAlignment.CENTER,
                                                spacing=10,
                                                controls=[
                                                    Text("1", size=50, weight="bold", color="#637aa7"),
                                                    Icon(
                                                        icons.UPLOAD_FILE,
                                                        size=50,
                                                        color="#637aa7",
                                                    ),
                                                    Text(
                                                        "Data Upload",
                                                        size=18,
                                                        weight="bold",
                                                        color="#000000",
                                                        text_align="center",
                                                    ),
                                                    Text(
                                                        "Login and Upload File",
                                                        size=14,
                                                        color="#555555",
                                                        text_align="center",
                                                    ),
                                                ],
                                            ),
                                        ),
                                        Container(
                                            border_radius=10,
                                            border=border.all(1, 'black12'),
                                            bgcolor="#dee2e6",
                                            width=170,
                                            height=300,
                                            padding=padding.all(10),
                                            content=Column(
                                                alignment=MainAxisAlignment.START,
                                                horizontal_alignment=CrossAxisAlignment.CENTER,
                                                spacing=10,
                                                controls=[
                                                    Text("2", size=50, weight="bold", color="#637aa7"),
                                                    Icon(
                                                        icons.COMPUTER_OUTLINED,
                                                        size=50,
                                                        color="#637aa7",
                                                    ),
                                                    Text(
                                                        "Deepfake Identification",
                                                        size=18,
                                                        weight="bold",
                                                        color="#000000",
                                                        text_align="center",
                                                    ),
                                                    Text(
                                                        "Recognize differences between real and fake content.",
                                                        size=14,
                                                        color="#555555",
                                                        text_align="center",
                                                    ),
                                                ],
                                            ),
                                        ),
                                        Container(
                                            bgcolor="#dee2e6",
                                            border_radius=10,
                                            border=border.all(1, 'black12'),
                                            width=170,
                                            height=300,
                                            padding=padding.all(10),
                                            content=Column(
                                                alignment=MainAxisAlignment.START,
                                                horizontal_alignment=CrossAxisAlignment.CENTER,
                                                spacing=10,
                                                controls=[
                                                    Text("3", size=50, weight="bold", color="#637aa7"),
                                                    Icon(
                                                        icons.VERIFIED,
                                                        size=50,
                                                        color="#637aa7",
                                                    ),
                                                    Text(
                                                        "Verification",
                                                        size=18,
                                                        weight="bold",
                                                        color="#000000",
                                                        text_align="center",
                                                    ),
                                                    Text(
                                                        "Get Result",
                                                        size=14,
                                                        color="#555555",
                                                        text_align="center",
                                                    ),
                                                ],
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                            alignment=MainAxisAlignment.START,
                            spacing=10,
                        ),
                    ),
                    Container(
                        height=550,
                        width=550,
                        padding=padding.all(20),
                        bgcolor="#f5f5f5",
                        border_radius=10,
                        border=border.all(1, 'black'),
                        content=Column(
                            alignment=MainAxisAlignment.START,
                            horizontal_alignment=CrossAxisAlignment.CENTER,
                            spacing=20,
                            controls=[
                                Row(
                                    alignment=MainAxisAlignment.CENTER,
                                    spacing=15,
                                    controls=[
                                        Icon(icons.IMAGE, size=32, color=colors.INDIGO_500),
                                        Icon(icons.VIDEO_LIBRARY, size=32, color=colors.INDIGO_500),
                                        Icon(icons.AUDIO_FILE, size=32, color=colors.INDIGO_500),
                                    ],
                                ),
                                Container(
                                    width = 450,
                                    padding=padding.all(5),
                                    border=border.all(1, colors.BLACK26),
                                    border_radius=5,
                                    content=Row(
                                        alignment=MainAxisAlignment.SPACE_BETWEEN,
                                        controls=[
                                            Text(
                                                "\t\t\t\t\tSelect a image or video or audio file",
                                                size=14,
                                                color="#666666",
                                            ),
                                            ElevatedButton(
                                                text="Browse",
                                                icon=icons.FOLDER_OPEN,
                                                on_click=self._upload_file,
                                                style=ButtonStyle(
                                                    bgcolor=colors.BLUE,
                                                    color=colors.WHITE,
                                                    shape=RoundedRectangleBorder(radius=5),
                                                ),
                                            ),
                                        ],
                                    ),
                                ),
                                Container(height=5),
                                ElevatedButton(
                                    text="Detect Now",
                                    icon = icons.BATCH_PREDICTION,
                                    on_click=self._call_detect_deepfake,
                                    style=ButtonStyle(
                                        bgcolor=colors.BLUE,
                                        color=colors.WHITE,
                                        shape=RoundedRectangleBorder(radius=5),
                                        #padding=padding.all(10),
                                    ),
                                ),
                                
                                Container(height=20),
                                Divider(height=1, color=colors.BLACK12),
                                
                                Row(
                                    alignment=MainAxisAlignment.START,
                                    controls=[
                                        Text(
                                            "Result :",
                                            size=25,
                                            weight="bold",
                                            color="#480a32",
                                            text_align="center",
                                        ),

                                    ]
                                ),
                                Column(
                                    alignment=MainAxisAlignment.CENTER,
                                    controls = [
                                        Container(content=self.t1,alignment=alignment.center),
                                        Container(content=self.progress_detect,alignment=alignment.center),
                                        Container(content=self.uploaded_file_path_text,alignment=alignment.center),
                                    ]
                                ),
                                Row(
                                    alignment=MainAxisAlignment.CENTER,
                                    controls=[
                                        self.file_result,
                                    ]
                                ),
                                
                            ],
                        ),
                    ),
                ],
                spacing=20,
            ),
        )




    def _upload_file(self, e):
        self.file_result.visible = False
        self.file_result.value = None
        self.uploaded_file_path_text.visible = False
        self.t1.visible = False
        self.progress_detect.visible = False
        file_picker = FilePicker(on_result=self._on_file_selected)
        self.page.overlay.append(file_picker)
        self.page.update()
        file_picker.pick_files(allow_multiple=False)

    def _on_file_selected(self, e):
        if e.files and len(e.files) > 0:
            self._file_path = e.files[0].path
            file_extension = os.path.splitext(self._file_path)[1].lower()
            if file_extension in self.allowed_extensions:
                self.selected_file_path = self._file_path

                #self.uploaded_file_path_text.value = f"Uploaded file: {_file_path}"
                self._show_upload_result(f"File uploaded successfully: {self._file_path} \nNow Click on Detect Now...!")
            else:
                #self.uploaded_file_path_text.value = "Invalid file type. Please upload an image, audio, or video file."
                self._show_upload_result("Error: Unsupported file type. \nPlease upload an image, audio, or video file.")
            self.page.update()
        else:
            self.uploaded_file_path_text.visible = True
            self.uploaded_file_path_text.value = "No file selected"
            self.page.update()

    def _show_upload_result(self, message):
        dialog = AlertDialog(
            title=Text("Upload Result"),
            content=Container(
                height=150,
                content=Column(
                    controls=[Text(message, size=16)],
                ),
            ),
            actions=[TextButton("OK", on_click=lambda e: self.on_success_dialog_dismiss(dialog))]
        )
        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def on_success_dialog_dismiss(self, dialog):
        dialog.open = False
        self.page.update()

    def _show_user_info(self, e):
        user_info = auth.current_user
        if user_info:
            email = user_info.get("email", "Unknown Email")
            uid = user_info.get("localId")
            try:
                user_data = database.child("users").child(uid).get()
                username = user_data.val().get("username", "Unknown User")
            except Exception:
                username = "Unknown User"
        else:
            email = "No user logged in"
            username = "N/A"
        dialog = AlertDialog(
            title=Text("User Info"),
            content=Container(
                height=150,
                content=Column(
                    controls=[Text(f"User Name: {username}", size=16), Text(f"Email: {email}", size=16)],
                ),
            ),
            actions=[TextButton("OK", on_click=lambda e: self.on_success_dialog_dismiss(dialog))]
        )
        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def _logout(self, e):
        auth.current_user = None
        self.page.go('/login')
        self.page.update()

    def _blog_link(self,e):
        webbrowser.open("https://www.ibm.com/blog/deepfake-detection/")

    def _article_link(self,e):
        webbrowser.open("https://en.wikipedia.org/wiki/Deepfake")

    def _git_link(self,e):
        webbrowser.open("https://github.com/dhananjaya2003/DeepScan_Pro_Deepfake-Detector")

    def _open_google(self, e):
        webbrowser.open("https://www.google.com")

    def _download_user_manual(self, e):
        file_path = r"C:\Users\hp\python_jupyter\TY_PROJECT\User Guide for DeepScan App.pdf"
        if os.path.exists(file_path):
            os.startfile(file_path)

    def _open_settings_dialog(self, e):
        theme_switch = Switch(
            value=self.page.theme_mode == ThemeMode.DARK,
            on_change=self._toggle_theme,
            label="Dark Mode",
        )
        dialog = AlertDialog(
            title=Text("Settings"),
            content=Container(
                height=100,
                content=Column(
                    controls=[theme_switch]
                ),
            ),
            actions=[TextButton("Close", on_click=lambda e: self.on_success_dialog_dismiss(dialog))]
        )
        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def _toggle_theme(self, e):
        self.page.theme_mode = ThemeMode.DARK if self.page.theme_mode == ThemeMode.LIGHT else ThemeMode.LIGHT
        self.page.update()

    def _send_bug_report(self, e):
        bug_report_email = "deepscanpro.deepdetetctor@gmail.com"
        subject = "Bug Report"
        body = "Please describe the bug here."
        mailto_link = f"https://mail.google.com/mail/?view=cm&fs=1&to={bug_report_email}&su={subject}&body={body}"
        webbrowser.open(mailto_link)

    def _call_detect_deepfake(self, e):
        if self.selected_file_path:
            self.uploaded_file_path_text.color = "#598392"
            self.uploaded_file_path_text.weight = FontWeight.BOLD
            self._detect_deepfake(self.selected_file_path)
            self.page.update()
        else:
            self._show_upload_result("Error: No file selected. Please upload a file first.")

    def _detect_deepfake(self,file_path):
        if not os.path.exists(file_path):
            self._show_upload_result("Error: File not found. Please re-upload.")
        else:
            self.progress_detect.visible = True
            self.t1.visible = True
            for i in range(0, 101):
                self.progress_detect.value = i * 0.01
                sleep(0.1)
                self.page.update()

            a, b = load_model_image()
            self.file_result.weight = FontWeight.BOLD
            self.file_result.size = 20
            
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
                self.res_temp = detect_image(file_path, a, b)
            elif file_extension in [".mp3", ".wav"]:
                self.res_temp = audio_deepfake(file_path)
            elif file_extension in [".mp4", ".avi", ".mkv", ".mov"]:
                self.res_temp = video_deepfake(file_path)
            else:
                self._show_upload_result("Error: Unsupported file type.")
                return

            if self.res_temp == "Fake":
                self.file_result.color = colors.RED_600
                self.file_result.value = "Deepfake Detected!⛔"
            elif self.res_temp == 'Real':
                self.file_result.color = colors.GREEN_600
                self.file_result.value = "No Deepfake Detected!✅"
            else:
                self.file_result.weight = None
                self.file_result.color = colors.GREEN_600
                self.file_result.value = "Sorry for inconvinence! \nVideo model is under development!"
                self.file_result.size = 10
                

            self.uploaded_file_path_text.value = f"File Selected: {os.path.basename(self.selected_file_path)}"
            self.uploaded_file_path_text.visible = True
            self.file_result.visible = True
            self.t1.visible = False
            self.page.update()



    
            
           
            
            
    


