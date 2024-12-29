from flet import *
import flet
import webbrowser


class StartPage(Container):
    def __init__(self, page: Page):
        super().__init__()
        self.page = page
        page.title = "DeepScan Pro"
        page.theme_mode = ThemeMode.LIGHT
        page.scroll = "always"
        page.fonts = {"Bebas Neue" : r"D:\Final Year Project\DeepScan_Pro\assets\fonts\BebasNeue-Regular.ttf",
                      "Neuton": r'D:\Final Year Project\DeepScan_Pro\assets\fonts\Neuton-Regular.ttf',
                      "Poppins" : r'D:\Final Year Project\DeepScan_Pro\assets\fonts\Poppins-ExtraBold.ttf',
                      "Cabin" : r"D:\Final Year Project\DeepScan_Pro\assets\fonts\Cabin-Regular.ttf"
                      }
        self.expand = True
        self.page.scroll = 'always'
        self.contact_ref = Ref[Container]()
        self.bgcolor = '#f5f7fd'
        self.images = [
            r"D:\Final Year Project\DeepScan_Pro\assets\images\images.jpeg",
            r"D:\Final Year Project\DeepScan_Pro\assets\images\deepfakes-threat-kyc-aml.png",
            r"D:\Final Year Project\DeepScan_Pro\assets\images\deepfakes-verification.png",
            r"D:\Final Year Project\DeepScan_Pro\assets\images\download.jpeg",
            r"D:\Final Year Project\DeepScan_Pro\assets\background.jpg"
        ]
        self.content = Column(
            expand=True,
            scroll=ScrollMode.ALWAYS,
            controls=[
                self._build_header_start(),
                Divider(height=1, color=colors.BLACK26),
                Row(
                    expand=True,
                    controls=[
                        self._build_sidebar_start(),
                        self._build_main_content_start(),
                    ],
                ),
                Container(height=20),
                Container(content = self._create_image_slider(self.images),
                          alignment=alignment.center,),
                Divider(),
                Container(height=50,
                        width = 200,
                        expand = True,
                        bgcolor='#adb5bd',
                        border_radius = 7,
                        padding=padding.only(left=50,right=50),
                        content = Row(controls =[
                                    Text(
                                        value ="Use Cases",
                                        size = 20,
                                        color=colors.BLACK,
                                        bgcolor = None,
                                        weight=FontWeight.BOLD,
                            ),
                            
                        ],
                            
                    )
                ),
                self._use_case(),
                Container(height=50),
                Divider(),
                Container(height=50,
                        width = 200,
                        expand = True,
                        bgcolor='#90e0ef',
                        border_radius = 7,
                        padding=padding.only(left=50,right=50),
                        content = Row(controls =[
                                    Text(
                                        value ="Developers",
                                        size = 20,
                                        color=colors.BLACK,
                                        bgcolor = None,
                                        weight=FontWeight.BOLD,
                            ),
                            
                        ],
                            
                    )
                ),
                self._dev_info(),
                Container(height=50),
                Divider(),
                Container(height=50,
                        width = 200,
                        bgcolor='#57cc99',
                        border_radius =7,
                        expand = True,
                        padding=padding.only(left=50,right=50),
                        content = Row(
                            controls =[
                                Text(
                                    value ="Contact Us",
                                    size = 20,
                                    color=colors.BLACK,
                                    bgcolor = None,
                                    weight=FontWeight.BOLD,
                            ),
                            
                        ],
                            
                    )
                ),
                self._contact_info(),
            ],
        )
        

        

    def _build_header_start(self):
        return Container(
            height=90,
            padding=padding.only(left=130, right=150),
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
                    
                    Row(width=525),

                    Row(
                        controls=[
                            Text(
                                value="Home",
                                color=colors.BLACK,
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

                            
                            TextButton(
                                    "Use cases",
                                    on_click=lambda _: self.content.scroll_to(key="use_cases",duration=500),
                                    content=Text(
                                        value="Use cases", weight=FontWeight.BOLD,size=20,color="black",bgcolor=None,
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
                                    PopupMenuItem(text="Developers",on_click=lambda _: self.content.scroll_to(key="dev_container",duration=500),),
                                    PopupMenuItem(text="Data Privacy",on_click=lambda _: self.page.go('/faq')),
                                ],
                            ),

                            ElevatedButton(
                                    text="Contact",
                                    color="white",
                                    bgcolor="#58b999",
                                    on_click=lambda _: self.content.scroll_to(key="contact_container",duration=500),
                                    content=Text(
                                        value="Contact", weight=FontWeight.BOLD,size=19,
                                    ),
                                    style=ButtonStyle(
                                        shape=RoundedRectangleBorder(radius=0),
                                        padding=padding.all(15),
 
                                    )
                            ),
                            
                        ],
                        spacing=35,
                    ),
                   
                ],
            ),
        )

    def _build_sidebar_start(self):
        return Container(
            width=130,
            bgcolor=None,
        )
    
    def _build_main_content_start(self):
        return Container(
            expand=True,
            bgcolor=None,
            content=Row(
                alignment=MainAxisAlignment.START,
                vertical_alignment=CrossAxisAlignment.START,
                controls=[
                    Container(
                        width=600,
                        padding=padding.only(top=90, left=0, bottom=50),
                        content=Column(
                            controls=[
                                Text(
                                    "Deepfake Detection Software",
                                    size=55,
                                    weight="bold",
                                    color="#23324e",
                                    font_family="Poppins",
                                    
                                ),
                                Container(height=20),
                                Text(
                                    "Reliable technology to detect \ndeepfakes & prevent spoofing",
                                    size=30,
                                    weight="bold",
                                    color="#1b1e27",
                                    font_family="Neuton"
                                ),
                                Container(height=20),
                                Text(
                                    "In the landscape of advanced digital manipulation, a reliable \ndeepfake detector is highly in demand. DeepScan Pro employs sophisticated \nalgorithms, leveraging technologies to verify the authenticity of visual \ncontent. This guarantees a strong defense against the growing threat of \ndeceptive media.",
                                    size=16,
                                    color="#23324e",
                                    text_align="start",
                                    weight="bold",
                                    font_family="Cabin"
                                ),
                                Container(height=50),
                                ElevatedButton(
                                    text="Try it now",
                                    color="white",
                                    bgcolor="#5f7ab0",
                                    on_click=lambda _: self.page.go("/login"),

                                    content=Text(
                                        value="Try it now", weight=FontWeight.BOLD
                                    ),
                                    style=ButtonStyle(
                                        shape=RoundedRectangleBorder(radius=5),
                                        padding=padding.only(left=40,right=40,top=20,bottom=20),
 
                                    )
                                ),
                            ],
                            alignment=MainAxisAlignment.START,
                            spacing=0,
                        ),
                    ),
                    Container(
                        height=700,
                        width=700,
                        padding=padding.only(right=40,top=0,bottom=50),
                        bgcolor=None,
                        content=Image(
                            src=r'D:\Final Year Project\DeepScan_Pro\assets\images\Deep-fake-detection-AI.jpg',
                            fit=ImageFit.CONTAIN,
                            width=650,
                            height=650,

                        )
                    ),
                ],
                spacing=20,
            ),   
        )
    
    def _blog_link(self,e):
        webbrowser.open("https://www.ibm.com/blog/deepfake-detection/")

    def _article_link(self,e):
        webbrowser.open("https://en.wikipedia.org/wiki/Deepfake")

    def _git_link(self,e):
        webbrowser.open("https://github.com/dhananjaya2003/DeepScan_Pro_Deepfake-Detector")

    
    def _contact_info(self):
        return Container(
            padding=padding.only(left =100,top=20),
            key="contact_container",
            content=Column(
                width=500,
                controls=[
                    Row(
                        controls=[
                            Icon(name=icons.EMAIL),
                            Text("deepscanpro.deepdetetctor@gmail.com", size=16,)
                        ],
                        alignment='start',
                    ),
                    Row(
                        controls=[
                            Icon(name=icons.PHONE),
                            Text("+123 456 7890", size=16)
                        ],
                        alignment="start",
                    ),
                    Row(
                        controls=[
                            Icon(name=icons.LOCATION_ON),
                            Text("D Y Patil College of Engineering & Technology, Kolhapur", size=16)
                        ],
                        alignment="start",
                    ),
                    Divider(thickness=2),
                    Row(
                        controls=[
                            Image(
                                src="https://cdn-icons-png.flaticon.com/512/174/174857.png",
                                width=30,
                                height=30,
                                tooltip="LinkedIn",
                            ),
                            Image(
                                src="https://cdn-icons-png.flaticon.com/128/4423/4423697.png",
                                width=30,
                                height=30,
                                tooltip="Whatsapp",
                            ),
                            Image(
                                src="https://cdn-icons-png.flaticon.com/128/5968/5968958.png",
                                width=30,
                                height=30,
                                tooltip="X (Twitter)",
                            ),
                            Image(
                                src="https://cdn-icons-png.flaticon.com/512/733/733553.png",
                                width=30,
                                height=30,
                                tooltip="GitHub",
                            ),
                        ],
                        alignment="start",
                        spacing=20,
                    ),
                ],
                spacing=15,
                alignment="center",
            ),
            expand=True,
        )
    
    def _dev_info(self):
        return Container(
                key="dev_container",
                content=Row(
                    controls=[
                        Column(
                            controls=[
                                Image(src="https://cdn-icons-png.flaticon.com/128/6997/6997662.png", width=150, height=150),
                                Text("\t\t\t\t\t\t\t\t\t\t\tShreya", size=16, weight="bold")
                            ],
                            alignment="center",
                        ),
                        Column(
                            controls=[
                                Image(src="https://cdn-icons-png.flaticon.com/128/6997/6997674.png", width=150, height=150),
                                Text("\t\t\t\t\t\t\t\tDhananjay", size=16, weight="bold")
                            ],
                            alignment="center",
                        ),
                        Column(
                            controls=[
                                Image(src="https://cdn-icons-png.flaticon.com/128/6997/6997674.png", width=150, height=150),
                                Text("\t\t\t\t\t\t\t\t\t\t\tManasvi", size=16, weight="bold")
                            ],
                            alignment="center",
                        ),
                        Column(
                            controls=[
                                Image(src="https://cdn-icons-png.flaticon.com/128/6997/6997674.png", width=150, height=150),
                                Text("\t\t\t\t\t\t\t\t\t\t\t\tAryan", size=16, weight="bold")
                            ],
                            alignment="center",
                        ),
                    ],
                    alignment="center",
                    spacing=30,
                ),
                padding=padding.only(top=50),
                expand=True,
            )
    
    def _use_case(self):
        row1 = Row(
            controls=[
                Column(
                    controls=[
                        Image(src="https://cdn-icons-png.flaticon.com/128/6784/6784655.png", width=180, height=180),
                        Text(value="Media Authenticity Verification", size=16,weight=FontWeight.BOLD,text_align='center',font_family='Cabin'),
                    ]
                ),
                Row(width=20),
                Column(
                    controls=[
                        Image(src="https://cdn-icons-png.flaticon.com/128/13083/13083198.png", width=180, height=180),
                        Text(value="Law Enforcement and Cybersecurity", size=16,weight=FontWeight.BOLD,text_align='center',font_family='Cabin'),
                    ]
                ),
                Row(width=20),
                Column(
                    controls=[
                        Image(src="https://cdn-icons-png.flaticon.com/128/16196/16196639.png", width=180, height=180),
                        Text(value="Government and Defense", size=16,weight=FontWeight.BOLD,text_align='center',font_family='Cabin'),
                    ]
                ),
                Row(width=20),
                Column(
                    controls=[
                        Image(src="https://cdn-icons-png.flaticon.com/128/1236/1236575.png", width=180, height=180),
                        Text(value="Education and Training Tools", size=16,weight=FontWeight.BOLD,text_align='center',font_family='Cabin'),
                    ]
                ),

                
            ],
            spacing=35,
            alignment="center",
        )

        row2 = Row(
            controls=[
                Column(
                    controls=[
                        Image(src="https://cdn-icons-png.flaticon.com/128/7356/7356470.png", width=180, height=180),
                        Text(value="Entertainment Industry", size=16,weight=FontWeight.BOLD,text_align='center',font_family='Cabin'),
                    ]
                ),
                Row(width=50),
                Column(
                    controls=[
                        Image(src="https://cdn-icons-png.flaticon.com/128/2382/2382533.png", width=180, height=180),
                        Text(value="Healthcare & Drug Discovery", size=16,weight=FontWeight.BOLD,text_align='center',font_family='Cabin'),
                    ]
                ),
                Row(width=60),
                Column(
                    controls=[
                        Image(src="https://cdn-icons-png.flaticon.com/128/11100/11100051.png", width=180, height=180),
                        Text(value="Consumer Applications", size=16,weight=FontWeight.BOLD,text_align='center',font_family='Cabin'),
                    ]
                ),
                Row(width=50),
                Column(
                    controls=[
                        Image(src="https://cdn-icons-png.flaticon.com/128/4187/4187272.png", width=180, height=180),
                        Text(value="Social Media Content Check", size=16,weight=FontWeight.BOLD,text_align='center',font_family='Cabin'),
                    ]
                )
                
            ],
            spacing=35,
            alignment="center",
        )

        return Container(
            key = 'use_cases',
            height=470,
            expand=True,
            padding=padding.only(left=70, right=50,bottom=50),
            content=Column(controls=[row1, row2], spacing=30, alignment="center"),
        )

    def _create_image_slider(self,images: list):
        current_index = 0
        def update_image(index: int):
            nonlocal current_index
            current_index = index
            img_control.src = images[current_index]
            self.page.update()

        def on_prev_click(e):
            new_index = current_index - 1 if current_index > 0 else len(images) - 1
            update_image(new_index)

        def on_next_click(e):
            new_index = current_index + 1 if current_index < len(images) - 1 else 0
            update_image(new_index)

        img_control = Image(src=images[current_index], fit=ImageFit.CONTAIN,width=400, height=300)

        prev_button = IconButton(icons.CHEVRON_LEFT, on_click=on_prev_click)
        next_button = IconButton(icons.CHEVRON_RIGHT, on_click=on_next_click)

        return Container(
            Row(
                [prev_button, img_control, next_button],
                alignment=MainAxisAlignment.CENTER,
                spacing=20
            ),
            border=border.all(2, colors.BLACK12), 
            border_radius=10,
            padding=20,
            margin=20,
            width=550,
            height=400,
            alignment=alignment.center
        )

           



   
            
            
    


