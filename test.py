s = "ada"


# def textCheck(text):
#     reversedString =''.join(reversed(text))
#     if text == reversedString:
#         print('ok')
#     else:
#         print('not a palindrome')

# textCheck("NisiOisiN")

l = ''.join([ s[x:x+2][::-1] for x in range(0, len(s), 2) ])
print(l)