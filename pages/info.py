from flet import *
import flet
from service_a.auth_user import *

class Contact(Container):
    def __init__(self, page: Page):
        super().__init__()
        self.page = page
        self.bgcolor = "#eef4ed"
        self.create_layout()

    def create_layout(self):
        dev_info = Container(
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
        

        # Contact Info Section
        contact_info = Container(
            padding=padding.only(left =100,top=20),
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

        self.content = Column(
            controls=[
                Container(height=50,
                        bgcolor='#90e0ef',
                        expand = True,
                        padding=padding.only(left=50),
                        content = Row(controls =[
                                                Text(
                                                    value ="Developers",
                                                    size = 20,
                                                    color=colors.BLACK,
                                                    bgcolor = None,
                                                    weight=FontWeight.BOLD,
                            ),
                            Row(width=400),
                            IconButton(
                                icon=icons.ARROW_LEFT_ROUNDED,
                                icon_color=colors.GREY_800,
                                icon_size=30,
                                bgcolor = colors.GREY_300, 
                                on_click=lambda _: self.page.go("/check") if auth.current_user else self.page.go("/startpage"),
                                style=ButtonStyle(
                                    CircleBorder(),
                                ),
                            ),

                        ],
                            
                    )
                ),       
                Container(content=dev_info, expand=True),
                Container(height=110),
                Container(height=50,
                        bgcolor='#57cc99',
                        expand = True,
                        padding=padding.only(left=50),
                        content = Row(controls =[
                                                Text(
                                                    value ="Contact Us",
                                                    size = 20,
                                                    color=colors.BLACK,
                                                    bgcolor = None,
                                                    weight=FontWeight.BOLD,
                            ),
                            Row(width=400),
                            IconButton(
                                icon=icons.ARROW_LEFT_ROUNDED,
                                icon_color=colors.GREY_800,
                                icon_size=30,
                                bgcolor = colors.GREY_300, 
                                on_click=lambda _: self.page.go("/check") if auth.current_user else self.page.go("/startpage"),
                                style=ButtonStyle(
                                    CircleBorder(),
                                ),
                            ),
                        ],
                            
                    )
                ),
                Container(content=contact_info, expand=True),
                Container(expand=True,bgcolor=None,height=100)
            ],
            expand=True,
        )



