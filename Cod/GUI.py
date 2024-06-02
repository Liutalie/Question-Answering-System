from tkinter import *
import InformationRetrieval
import TextProcessing
import AnswerExtraction
import main


class GUI(Frame):
    def __init__(self, master):
        super(GUI, self).__init__(master)
        self.createWidgets()

    def searchButtonAction(self):
        infoRetrieval = InformationRetrieval.InformationRetrieval()
        textProcessing = TextProcessing.TextProcessing()
        answerExtraction = AnswerExtraction.AnswerExtraction()
        textbox = self.nametowidget('questinEntry')
        question = textbox.get('1.0', 'end-1c')
        answer = main.getAnswer(question, infoRetrieval, textProcessing, answerExtraction)
        answer_string = ''
        answer_textbox = self.nametowidget('answerTextbox')
        answer_textbox.config(state='normal')
        answer_textbox.delete('1.0', END)
        for elem in answer:
            answer_string += str(elem)
            if len(answer) == 2 and answer.index(elem) == 0:
                answer_string += '\n'
        answer_textbox.insert(END, answer_string)
        answer_textbox.config(state='disabled')


    def createWidgets(self):
        self.label = Label(self, text="What is your question?", font=("Times New Roman", 16), name='questionLabel')
        self.label.place(x=20, y=20)

        self.textbox = Text(self, width=120, height=1, font=("Times New Roman", 14), name='questinEntry')
        self.textbox.place(x=20, y=50)

        self.button = Button(self, text="Search", command=self.searchButtonAction, name='searchButton')
        self.button.place(x=1110, y=47, height=30, width=80)

        self.label = Label(self, text="Answer", font=("Times New Roman", 16), name='answerLabel')
        self.label.place(x=20, y=100)

        self.answer_textbox = Text(self, width=120, height=1, font=("Times New Roman", 14), name='answerTextbox')
        self.answer_textbox.place(x=20, y=130, height=300)
