from script_final import language_classifier

print('\n')
home_question = int(input('Deseja digitar quantas frases: '))

index = 0
data = []

for c in range(0,home_question):
    index += 1
    sentence = input(f'\nDigite a {index}º frase desejada: ')
    data.append(sentence)

print('\n')
language_classifier(data)