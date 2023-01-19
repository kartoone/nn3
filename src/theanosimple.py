# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

#a = 3
#b = 4
#c = a + b
#print(a)
#print(b)
#print(c)

a = T.dscalar()
b = T.dscalar()
c = T.dscalar()

def ouroutputfunc():
    return a+b+c

#d = a+b+c
f = theano.function([a,b,c], ouroutputfunc())
print(f(3, 4, 5))
print(f(5, 6, 7))


