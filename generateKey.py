from AESCipher import AESCipher
import sys
import argparse
'''
Example
python generateKey -e "192 wrwfs" 
python generateKey.py -e "gAAAAABjys268_1msg4lb7b_LJAhngvUeRS1Ns_NWPK8U2y1P06gXqjC2H8Z2yf0qeVmr5pdHAoEoDxVFJI7B2aiF1Me0UIfmQ==" 
'''

c = AESCipher()
command = {'encrypt':c.encrypt, 'decrypt':c.decrypt}
 
parser = argparse.ArgumentParser(description="Encrypt/Decrypt the string specified."
                                                +"eg.: python generateKey.py -e '192 wrwfs'",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-e", "--encrypt",  help="Encryption")
parser.add_argument("-d", "--decrypt",  help="Decryption")

args = parser.parse_args()
config = vars(args)

for x in command:
    if x in config and config[x]:
        print(str(command[x](config[x].strip())))

    