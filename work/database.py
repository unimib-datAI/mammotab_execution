from mongoengine import connect
from mongoengine import Document, StringField, IntField, BooleanField, FloatField


class Cea(Document):
    model = StringField()
    prompt = StringField()
    cell = StringField()
    model_response = StringField()
    correct_response = StringField()
    table = StringField()
    row = IntField()
    column = IntField()
    correct = BooleanField()
    avg_time = FloatField()


class Missings(Document):
    cell = StringField()
    table = StringField()
    row = IntField()
    column = IntField()


class Database:

    def __init__(self):
        connect('mammotab', host='mongo', port=27017, username='root',
                password='mammotab_execution', authentication_source='admin')

    def save_missings(self, cell: str, table: str, row: int, column: int):
        Missings(cell=cell, table=table, row=row, column=column).save()

    def save_response(self, model_name: str, prompt: str, cell: str, model_response: str, correct_response: str, table: str, row: int, column: int, correct: bool, avg_time: float):
        Cea(
            model=model_name,
            prompt=prompt,
            cell=cell,
            model_response=model_response,
            correct_response=correct_response,
            table=table,
            row=row,
            column=column,
            correct=correct,
            avg_time=avg_time
        ).save()

    def get_all_documents(self):
        return Cea.objects
