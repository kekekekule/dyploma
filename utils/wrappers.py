from .colors import ColorPrint as colors


class JobWithMessage:
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        print(self.message, end=" ")

    def __exit__(self, *args):
        print(colors.format("OK", colors.OKGREEN))
