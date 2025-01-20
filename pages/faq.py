from flet import *
import flet
from service_a.auth_user import *

class PolicyFAQ(Container):
    def __init__(self,page:Page):
        super().__init__()
        self.page = page
        self.bgcolor = '#eef4ed'
        self.build()


    def build(self):
        data_privacy_policy_section = Container(
            content=Column(
                controls=[
                    Text("Data Privacy Policy", size=24, weight="bold"),
                    Text(
                        "At DeepScan Pro, we value your privacy and are committed to protecting the personal information you share with us. "
                        "This Privacy Policy outlines how we collect, use, and safeguard your data when you interact with our website and services.",
                        size=18,
                        selectable=True
                    ),
                    Text(
                        "The collected information is used to:\n"
                        "• Deliver and improve our services.\n"
                        "• Enhance your user experience with personalized content.\n"
                        "• Ensure the security of our platform.\n"
                        "• Communicate updates, respond to inquiries, and provide support.",
                        size=18,
                        selectable=True
                    ),
                ],
                spacing=10
            ),
            padding=padding.only(left=40, right=30, top=30),
            border_radius=10,
            border=border.all(1, colors.BLACK12),
            bgcolor="None",
            alignment=alignment.center,
            width=700,
            height=350
        )

        faq_section = Container(
            padding=padding.only(left=150, top=50),
            content=Column(
                controls=[
                    Row(
                        controls=[
                            Text("What is Deepfake Detection?", size=18, weight="bold", color="#192a51"),
                            Row(width=345),
                            Container(
                                content=Text(
                                    "Deepfake detection technology involves identifying AI-generated "
                                    "content, such as realistic videos or images, created by using deep "
                                    "learning techniques. Methods include forensic analysis, biometric "
                                    "comparison, machine learning models, and specialized tools. These "
                                    "methods distinguish between authentic and manipulated fake media. "
                                    "As deepfake technology advances, efforts aim to detect misuse and "
                                    "maintain trust in multimedia.",
                                    size=16,
                                    color=colors.WHITE,
                                ),
                                bgcolor="#192a51",
                                padding=15,
                                width=600,
                                border_radius=10,
                            ),
                        ],
                        spacing=10,
                        alignment="start",
                    ),
                    Row(
                        controls=[
                            Text("Why Deepfake Detection?", size=18, weight="bold", color="#192a51"),
                            Row(width=370),
                            Container(
                                content=Text(
                                    "Deepfake detection systems are the best way to combat malicious "
                                    "AI-generated media content. By implementing deepfake identification, "
                                    "individuals and organizations can minimize the risks of misinformation, "
                                    "privacy breaches, and potential manipulation of public opinion. This "
                                    "maintains the integrity of and trust in the digital world.",
                                    size=16,
                                    color=colors.WHITE,
                                ),
                                bgcolor="#192a51",
                                padding=15,
                                width=600,
                                border_radius=10,
                            ),
                        ],
                        spacing=10,
                        alignment="start",
                    ),
                    Row(
                        controls=[
                            Text("What are the benefits of deepfake detection tools?", size=18, weight="bold", color="#192a51"),
                            Row(width=155),
                            Container(
                                content=Text(
                                    "Voices and likenesses developed using deepfake technology can be used "
                                    "in movies to achieve a creative effect or maintain a cohesive story when "
                                    "the entertainers themselves are not available.",
                                    size=16,
                                    color=colors.WHITE,
                                ),
                                bgcolor="#192a51",
                                padding=15,
                                width=600,
                                border_radius=10,
                            ),
                        ],
                        spacing=10,
                        alignment="start",
                    ),
                    Row(
                        controls=[
                            Text("How can organizations prepare for deepfake threats?", size=18, weight="bold", color="#192a51"),
                            Row(width=135),
                            Container(
                                content=Text(
                                    "Implementing real-time verification for sensitive communications.\n"
                                    "Training personnel to recognize and respond to deepfakes.\n"
                                    "Protecting public data of high-priority individuals using watermarks.",
                                    size=16,
                                    color=colors.WHITE,
                                ),
                                bgcolor="#192a51",
                                padding=15,
                                width=600,
                                border_radius=10,
                            ),
                        ],
                        spacing=10,
                        alignment="start",
                    ),
                    Row(
                        controls=[
                            Text("What are examples of deepfake threats?", size=18, weight="bold", color="#192a51"),
                            Row(width=245),
                            Container(
                                content=Text(
                                    "Fake videos of public figures spreading misinformation.\n"
                                    "Impersonation of executives to manipulate stock prices or authorize fraudulent financial transactions.\n"
                                    "Synthetic media used during remote job interviews for fraudulent purposes.",
                                    size=16,
                                    color=colors.WHITE,
                                ),
                                bgcolor="#192a51",
                                padding=15,
                                width=600,
                                border_radius=10,
                            ),
                        ],
                        spacing=10,
                        alignment="start",
                    ),
                    Row(
                        controls=[
                            Text("Why are deepfakes a growing concern?", size=18, weight="bold", color="#192a51"),
                            Row(width=255),
                            Container(
                                content=Text(
                                    "Advances in AI have made tools more accessible and affordable, enabling "
                                    "even low-skilled actors to create sophisticated fakes.",
                                    size=16,
                                    color=colors.WHITE,
                                ),
                                bgcolor="#192a51",
                                padding=15,
                                width=600,
                                border_radius=10,
                            ),
                        ],
                        spacing=10,
                        alignment="start",
                    ),
                    Row(
                        controls=[
                            Text("Can your system detect deepfakes in real-time?", size=18, weight="bold", color="#192a51"),
                            Row(width=185),
                            Container(
                                content=Text(
                                    "Yes, DeepScan Pro detects deepfakes in real-time for images, videos & audio and "
                                    "provides quick results based on their size and resolution. Its design ensures efficient "
                                    "processing for immediate feedback.",
                                    size=16,
                                    color=colors.WHITE,
                                ),
                                bgcolor="#192a51",
                                padding=15,
                                width=600,
                                border_radius=10,
                            ),
                        ],
                        spacing=10,
                        alignment="start",
                    ),
                    Row(
                        controls=[
                            Text("What types of media does your system analyze?", size=18, weight="bold", color="#192a51"),
                            Row(width=180),
                            Container(
                                content=Text(
                                    "DeepScan Pro analyzes images, videos, and audio to detect manipulations. "
                                    "It identifies deepfake alterations in visual media and detects cloned or manipulated voices "
                                    "in audio files, providing a comprehensive detection solution.",
                                    size=16,
                                    color=colors.WHITE,
                                ),
                                bgcolor="#192a51",
                                padding=15,
                                width=600,
                                border_radius=10,
                            ),
                        ],
                        spacing=10,
                        alignment="start",
                    ),
                    Row(
                        controls=[
                            Text("What is the difference between deepfakes and cheap fakes?", size=18, weight="bold", color="#192a51"),
                            Row(width=80),
                            Container(
                                content=Text(
                                    "Deepfakes use AI/ML to create or alter content, while cheap fakes involve simpler techniques "
                                    "like altering playback speed or using basic editing tools without machine learning.",
                                    size=16,
                                    color=colors.WHITE,
                                ),
                                bgcolor="#192a51",
                                padding=15,
                                width=600,
                                border_radius=10,
                            ),
                        ],
                        spacing=10,
                        alignment="start",
                    ),
                ],
                spacing=20,
            ),
            alignment=alignment.center,
        )

        self.content = Column(
            controls=[
                Container(height=30),
                data_privacy_policy_section,
                Container(height=30),
                Container(height=50,
                          bgcolor='#d5c6e0',
                          expand=True,
                          padding=padding.only(left=50),
                          content=Row(controls=[
                              Text(
                                  value="FAQ's",
                                  size=20,
                                  color=colors.BLACK,
                                  bgcolor=None,
                                  weight=FontWeight.BOLD,
                              ),
                              Row(width=550),
                              IconButton(
                                  icon=icons.HOME_FILLED,
                                  icon_color=colors.GREY_800,
                                  icon_size=23,
                                  bgcolor=colors.GREY_300,
                                  on_click=lambda _: self.page.go("/check") if auth.current_user else self.page.go("/startpage"),
                                  style=ButtonStyle(
                                      CircleBorder(),
                                  ),
                              ),
                              Text(
                                  value="Back to Home",
                                  size=10,
                                  color=colors.BLACK,
                                  bgcolor=None,
                                  weight=FontWeight.BOLD,
                              ),
                          ],
                          )
                          ),
                faq_section,
            ],
            spacing=5,
            horizontal_alignment="center"
        )


