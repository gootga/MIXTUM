from PySide6.QtCore import QObject, Signal


class LogSystem(QObject):
    changed = Signal(str)

    def __init__(self, keys):
        QObject.__init__(self)

        self.keys = keys
        self.block = {}

        for key in keys:
            self.block[key] = ['']

        self.text = ''
        self.blocks = ['']

    def clear_entry(self, key):
        if key in self.block:
            for index in range(len(self.block[key])):
                self.block[key][index] = ''
            self.set_block()
            self.set_text()

    def append_entry(self, key, text):
        if key in self.block:
            self.block[key].append(text)
            self.set_block()
            self.set_text()

    def append_block(self):
        for key in self.block:
            self.block[key] = []
        self.blocks.append('')

    def set_text(self):
        self.text = '\n'.join(self.blocks)
        self.changed.emit(self.text)

    def set_block(self):
        text_lines = [line for key, lines in self.block.items() for line in lines if len(line) > 0]
        self.blocks[-1] = '\n'.join(text_lines)

    def set_entry(self, key, text, index = -1):
        if key in self.block and index < len(self.block[key]):
            self.block[key][index] = text
            self.set_block()
            self.set_text()
