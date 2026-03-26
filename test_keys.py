import webview
from webview.menu import Menu, MenuAction

def do_nothing():
    pass

menu_items = [
    Menu('File', [
        MenuAction('Quit', window_quit_doesntexist_just_testing, shortcut='cmd+q')
    ]),
    Menu('Edit', [
        MenuAction('Copy', do_nothing, shortcut='cmd+c'),
        MenuAction('Paste', do_nothing, shortcut='cmd+v'),
        MenuAction('Cut', do_nothing, shortcut='cmd+x'),
        MenuAction('Select All', do_nothing, shortcut='cmd+a'),
    ])
]

if __name__ == '__main__':
    window = webview.create_window('Test', html='<input type="text" value="Try to copy this text! Then paste it.">', frameless=True)
    webview.start(menu=menu_items)
