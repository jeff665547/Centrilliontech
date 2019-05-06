@ECHO ON
net use X: \\192.168.200.200\smtdata
x:
cd X:\jeff\Microarray\BanffC88NPAccuracy
python NP.py
net use X: /delete /y
@ECHO All analysis finished!
