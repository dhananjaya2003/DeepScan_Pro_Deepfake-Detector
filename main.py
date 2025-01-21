from flet import *
from pages.startpage import StartPage
from pages.login import Login
from pages.signup import Signup
from pages.check import HomePage
from pages.forgotpassword import Forgotpass
from pages.faq import PolicyFAQ
from pages.info import Contact
from util.colors import *


class Main(UserControl):
    def __init__(self,page:Page):
        super().__init__()
        self.page = page
        self.init_helper()

    def init_helper(self,):
        self.page.on_route_change = self.on_route_change
        self.page.go('/startpage')

    def on_route_change(self,route):
        new_page = {
            "/startpage":StartPage,
            "/login":Login,
            "/signup":Signup,
            "/forgotpassword":Forgotpass,
            "/check":HomePage,
            "/info":Contact,
            "/faq":PolicyFAQ
        }[self.page.route](self.page)

        self.page.views.clear()
        scroll_setting = None if route in ["/login", "/signup",'/check'] else "auto"
        self.page.views.append(
            View(
                route,
                [new_page],
                scroll=scroll_setting,
                
            )
        )
        self.page.update()



app(target=Main,assets_dir=r'/assets')
