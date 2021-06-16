import sys
import subprocess
import os
import json
import importlib
try:
    from importlib import metadata
except ImportError: # for Python<3.8
    import importlib_metadata as metadata
import re


def install(package):
    packageTest=re.sub('==.*','',package)
    packageTest=packageTest.lower()
    try : 
       importPackage = importlib.import_module(packageTest)
    except ModuleNotFoundError:
       print('Installing {} ...'.format(package))
       subprocess.check_call([sys.executable, "-m", "pip", "install", package])
       print('Successfull installed {}\n'.format(package))

    else :
        print(package)
        requestVersion=re.findall('==.*',package)
        if requestVersion!=[]:
          requestVersion=re.sub('==','',requestVersion[0])

        installVersion=metadata.version(packageTest)

        if str(requestVersion)==str(installVersion):
            print(package,' exist\n')
        else:
            print('Installing {} previously install version {} ...'.format(package,installVersion))
            try:
              subprocess.check_call([sys.executable, "-m", "pip", "install","--upgrade",'--force-reinstall',package])
            except Exception as e:
              print('Fail to install package {}. \nReason : {}\n'.format(package,e))
              flexiblePackage=re.findall('.*==',package)
              flexiblePackage=re.sub('==','',flexiblePackage[0])
              print('Executing flexible installation of ',flexiblePackage)
              answer=input('Flexible installation can cause a possibility of package conflict. Proceed?(y/n) ')
              if answer.lower()=='y':
                subprocess.check_call([sys.executable, "-m", "pip", "install","--upgrade",'--force-reinstall',flexiblePackage])
                print('Install ',flexiblePackage,' version ',version(flexiblePackage),' success\n')
              else:
                print()
            else:
              print('Successfull installed {}\n'.format(package))
              subprocess.check_call([sys.executable, "-m", "pip", "install","--upgrade",'--force-reinstall',flexiblePackage])


def downloadDialoGPT(path,modelName):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("microsoft/"+modelName)
    tokenizer.save_pretrained(path)
    
    model = AutoModelForCausalLM.from_pretrained("microsoft/"+modelName)
    model.save_pretrained(path)


def downloadDialoGRPT(path,modelName):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained("microsoft/"+modelName)
    tokenizer.save_pretrained(path)
    
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/"+modelName)
    model.save_pretrained(path)

def generateBadWordCorpus(pathBadWords):
    badWords = ['abbo', 'abo', 'abortion', 'abuse', 'addict', 'addicts', 'adult', 'africa', 'african', 'alla', 'alligatorbait', 'amateur', 'american', 'anal', 'analannie', 'analsex', 'angie', 'angry', 'anus', 'arab', 'arabs', 'areola', 'argie', 'aroused', 'arse', 'arsehole', 'asian', 'ass', 'assassin', 'assassinate', 'assassination', 'assault', 'assbagger', 'assblaster', 'assclown', 'asscowboy', 'asses', 'assfuck', 'assfucker', 'asshat', 'asshole', 'assholes', 'asshore', 'assjockey', 'asskiss', 'asskisser', 'assklown', 'asslick', 'asslicker', 'asslover', 'assman', 'assmonkey', 'assmunch', 'assmuncher', 'asspacker', 'asspirate', 'asspuppies', 'assranger', 'asswhore', 'asswipe', 'athletesfoot', 'attack', 'australian', 'babe', 'babies', 'backdoor', 'backdoorman', 'backseat', 'badfuck', 'balllicker', 'balls', 'ballsack', 'banging', 'baptist', 'barelylegal', 'barf', 'barface', 'barfface', 'bast', 'bastard ', 'bazongas', 'bazooms', 'beaner', 'beast', 'beastality', 'beastial', 'beastiality', 'beatoff', 'beat-off', 'beatyourmeat', 'beaver', 'bestial', 'bestiality', 'bi', 'biatch', 'bible', 'bicurious', 'bigass', 'bigbastard', 'bigbutt', 'bigger', 'bisexual', 'bi-sexual', 'bitch', 'bitcher', 'bitches', 'bitchez', 'bitchin', 'bitching', 'bitchslap', 'bitchy', 'biteme', 'black', 'blackman', 'blackout', 'blacks', 'blind', 'blow', 'blowjob', 'boang', 'bogan', 'bohunk', 'bollick', 'bollock', 'bomb', 'bombers', 'bombing', 'bombs', 'bomd', 'bondage', 'boner', 'bong', 'boob', 'boobies', 'boobs', 'booby', 'boody', 'boom', 'boong', 'boonga', 'boonie', 'booty', 'bootycall', 'bountybar', 'bra', 'brea5t', 'breast', 'breastjob', 'breastlover', 'breastman', 'brothel', 'bugger', 'buggered', 'buggery', 'bullcrap', 'bulldike', 'bulldyke', 'bullshit', 'bumblefuck', 'bumfuck', 'bunga', 'bunghole', 'buried', 'burn', 'butchbabes', 'butchdike', 'butchdyke', 'butt', 'buttbang', 'butt-bang', 'buttface', 'buttfuck', 'butt-fuck', 'buttfucker', 'butt-fucker', 'buttfuckers', 'butt-fuckers', 'butthead', 'buttman', 'buttmunch', 'buttmuncher', 'buttpirate', 'buttplug', 'buttstain', 'byatch', 'cacker', 'cameljockey', 'cameltoe', 'canadian', 'cancer', 'carpetmuncher', 'carruth', 'catholic', 'catholics', 'cemetery', 'chav', 'cherrypopper', 'chickslick', "children's", 'chin', 'chinaman', 'chinamen', 'chinese', 'chink', 'chinky', 'choad', 'chode', 'christ', 'christian', 'church', 'cigarette', 'cigs', 'clamdigger', 'clamdiver', 'clit', 'clitoris', 'clogwog', 'cocaine', 'cock', 'cockblock', 'cockblocker', 'cockcowboy', 'cockfight', 'cockhead', 'cockknob', 'cocklicker', 'cocklover', 'cocknob', 'cockqueen', 'cockrider', 'cocksman', 'cocksmith', 'cocksmoker', 'cocksucer', 'cocksuck ', 'cocksucked ', 'cocksucker', 'cocksucking', 'cocktail', 'cocktease', 'cocky', 'cohee', 'coitus', 'color', 'colored', 'coloured', 'commie', 'communist', 'condom', 'conservative', 'conspiracy', 'coolie', 'cooly', 'coon', 'coondog', 'copulate', 'cornhole', 'corruption', 'cra5h', 'crabs', 'crack', 'crackpipe', 'crackwhore', 'crack-whore', 'crap', 'crapola', 'crapper', 'crappy', 'crash', 'creamy', 'crime', 'crimes', 'criminal', 'criminals', 'crotch', 'crotchjockey', 'crotchmonkey', 'crotchrot', 'cum', 'cumbubble', 'cumfest', 'cumjockey', 'cumm', 'cummer', 'cumming', 'cumquat', 'cumqueen', 'cumshot', 'cunilingus', 'cunillingus', 'cunn', 'cunnilingus', 'cunntt', 'cunt', 'cunteyed', 'cuntfuck', 'cuntfucker', 'cuntlick ', 'cuntlicker ', 'cuntlicking ', 'cuntsucker', 'cybersex', 'cyberslimer', 'dago', 'dahmer', 'dammit', 'damn', 'damnation', 'damnit', 'darkie', 'darky', 'datnigga', 'dead', 'deapthroat', 'death', 'deepthroat', 'defecate', 'dego', 'demon', 'deposit', 'desire', 'destroy', 'deth', 'devil', 'devilworshipper', 'dick', 'dickbrain', 'dickforbrains', 'dickhead', 'dickless', 'dicklick', 'dicklicker', 'dickman', 'dickwad', 'dickweed', 'diddle', 'die', 'died', 'dies', 'dike', 'dildo', 'dingleberry', 'dink', 'dipshit', 'dipstick', 'dirty', 'disease', 'diseases', 'disturbed', 'dive', 'dix', 'dixiedike', 'dixiedyke', 'doggiestyle', 'doggystyle', 'dong', 'doodoo', 'doo-doo', 'doom', 'dope', 'dragqueen', 'dragqween', 'dripdick', 'drug', 'drunk', 'drunken', 'dumb', 'dumbass', 'dumbbitch', 'dumbfuck', 'dyefly', 'dyke', 'easyslut', 'eatballs', 'eatme', 'eatpussy', 'ecstacy', 'ejaculate', 'ejaculated', 'ejaculating ', 'ejaculation', 'enema', 'enemy', 'erect', 'erection', 'ero', 'escort', 'ethiopian', 'ethnic', 'european', 'evl', 'excrement', 'execute', 'executed', 'execution', 'executioner', 'explosion', 'facefucker', 'faeces', 'fag', 'fagging', 'faggot', 'fagot', 'failed', 'failure', 'fairies', 'fairy', 'faith', 'fannyfucker', 'fart', 'farted ', 'farting ', 'farty ', 'fastfuck', 'fat', 'fatah', 'fatass', 'fatfuck', 'fatfucker', 'fatso', 'fckcum', 'fear', 'feces', 'felatio ', 'felch', 'felcher', 'felching', 'fellatio', 'feltch', 'feltcher', 'feltching', 'fetish', 'fight', 'filipina', 'filipino', 'fingerfood', 'fingerfuck ', 'fingerfucked ', 'fingerfucker ', 'fingerfuckers', 'fingerfucking ', 'fire', 'firing', 'fister', 'fistfuck', 'fistfucked ', 'fistfucker ', 'fistfucking ', 'fisting', 'flange', 'flasher', 'flatulence', 'floo', 'flydie', 'flydye', 'fok', 'fondle', 'footaction', 'footfuck', 'footfucker', 'footlicker', 'footstar', 'fore', 'foreskin', 'forni', 'fornicate', 'foursome', 'fourtwenty', 'fraud', 'freakfuck', 'freakyfucker', 'freefuck', 'fu', 'fubar', 'fuc', 'fucck', 'fuck', 'fucka', 'fuckable', 'fuckbag', 'fuckbuddy', 'fucked', 'fuckedup', 'fucker', 'fuckers', 'fuckface', 'fuckfest', 'fuckfreak', 'fuckfriend', 'fuckhead', 'fuckher', 'fuckin', 'fuckina', 'fucking', 'fuckingbitch', 'fuckinnuts', 'fuckinright', 'fuckit', 'fuckknob', 'fuckme ', 'fuckmehard', 'fuckmonkey', 'fuckoff', 'fuckpig', 'fucks', 'fucktard', 'fuckwhore', 'fuckyou', 'fudgepacker', 'fugly', 'fuk', 'fuks', 'funeral', 'funfuck', 'fungus', 'fuuck', 'gangbang', 'gangbanged ', 'gangbanger', 'gangsta', 'gatorbait', 'gay', 'gaymuthafuckinwhore', 'gaysex ', 'geez', 'geezer', 'geni', 'genital', 'german', 'getiton', 'gin', 'ginzo', 'gipp', 'girls', 'givehead', 'glazeddonut', 'gob', 'god', 'godammit', 'goddamit', 'goddammit', 'goddamn', 'goddamned', 'goddamnes', 'goddamnit', 'goddamnmuthafucker', 'goldenshower', 'gonorrehea', 'gonzagas', 'gook', 'gotohell', 'goy', 'goyim', 'greaseball', 'gringo', 'groe', 'gross', 'grostulation', 'gubba', 'gummer', 'gun', 'gyp', 'gypo', 'gypp', 'gyppie', 'gyppo', 'gyppy', 'hamas', 'handjob', 'hapa', 'harder', 'hardon', 'harem', 'headfuck', 'headlights', 'hebe', 'heeb', 'hell', 'henhouse', 'heroin', 'herpes', 'heterosexual', 'hijack', 'hijacker', 'hijacking', 'hillbillies', 'hindoo', 'hiscock', 'hitler', 'hitlerism', 'hitlerist', 'hiv', 'ho', 'hobo', 'hodgie', 'hoes', 'hole', 'holestuffer', 'homicide', 'homo', 'homobangers', 'homosexual', 'honger', 'honk', 'honkers', 'honkey', 'honky', 'hook', 'hooker', 'hookers', 'hooters', 'hore', 'hork', 'horn', 'horney', 'horniest', 'horny', 'horseshit', 'hosejob', 'hoser', 'hostage', 'hotdamn', 'hotpussy', 'hottotrot', 'hummer', 'husky', 'hussy', 'hustler', 'hymen', 'hymie', 'iblowu', 'idiot', 'ikey', 'illegal', 'incest', 'insest', 'intercourse', 'interracial', 'intheass', 'inthebuff', 'israel', 'israeli', "israel's", 'italiano', 'itch', 'jackass', 'jackoff', 'jackshit',  'jade', 'jap', 'japcrap', 'jebus', 'jeez', 'jerkoff', 'jesus', 'jesuschrist', 'jew', 'jewish', 'jiga', 'jigaboo', 'jigg', 'jigga', 'jiggabo', 'jigger ', 'jiggy', 'jihad', 'jijjiboo', 'jimfish', 'jism', 'jiz ', 'jizim', 'jizjuice', 'jizm ', 'jizz', 'jizzim', 'jizzum', 'joint', 'juggalo', 'jugs', 'junglebunny', 'kaffer', 'kaffir', 'kaffre', 'kafir', 'kanake', 'kid', 'kigger', 'kike', 'kill', 'killed', 'killer', 'killing', 'kills', 'kink', 'kinky', 'kissass', 'kkk', 'knife', 'knockers', 'kock', 'kondum', 'koon', 'kotex', 'krap', 'krappy', 'kraut', 'kum', 'kumbubble', 'kumbullbe', 'kummer', 'kumming', 'kumquat', 'kums', 'kunilingus', 'kunnilingus', 'kunt', 'ky', 'kyke', 'lactate', 'laid', 'lapdance', 'latin', 'lesbain', 'lesbayn', 'lesbian', 'lesbin', 'lesbo', 'lez', 'lezbe', 'lezbefriends', 'lezbo', 'lezz', 'lezzo', 'liberal', 'libido', 'licker', 'lickme', 'lies', 'limey', 'limpdick', 'limy', 'lingerie', 'liquor', 'livesex', 'loadedgun', 'lolita', 'looser', 'loser', 'lotion', 'lovebone', 'lovegoo', 'lovegun', 'lovejuice', 'lovemuscle', 'lovepistol', 'loverocket', 'lowlife', 'lsd', 'lubejob', 'lucifer', 'luckycammeltoe', 'lugan', 'lynch', 'macaca', 'mad', 'mafia', 'magicwand', 'mams', 'manhater', 'manpaste', 'marijuana', 'mastabate', 'mastabater', 'masterbate', 'masterblaster', 'mastrabator', 'masturbate', 'masturbating', 'mattressprincess', 'meatbeatter', 'meatrack', 'meth', 'mexican', 'mgger', 'mggor', 'mickeyfinn', 'mideast', 'milf', 'minority', 'mockey', 'mockie', 'mocky', 'mofo', 'moky', 'moles', 'molest', 'molestation', 'molester', 'molestor', 'moneyshot', 'mooncricket', 'mormon', 'moron', 'moslem', 'mosshead', 'mothafuck', 'mothafucka', 'mothafuckaz', 'mothafucked ', 'mothafucker', 'mothafuckin', 'mothafucking ', 'mothafuckings', 'motherfuck', 'motherfucked', 'motherfucker', 'motherfuckin', 'motherfucking', 'motherfuckings', 'motherlovebone', 'muff', 'muffdive', 'muffdiver', 'muffindiver', 'mufflikcer', 'mulatto', 'muncher', 'munt', 'murder', 'murderer', 'muslim', 'naked', 'narcotic', 'nasty', 'nastybitch', 'nastyho', 'nastyslut', 'nastywhore', 'nazi', 'necro', 'negro', 'negroes', 'negroid', "negro's", 'nig', 'niger', 'nigerian', 'nigerians', 'nigg', 'nigga', 'niggah', 'niggaracci', 'niggard', 'niggarded', 'niggarding', 'niggardliness', "niggardliness's", 'niggardly', 'niggards', "niggard's", 'niggaz', 'nigger', 'niggerhead', 'niggerhole', 'niggers', "nigger's", 'niggle', 'niggled', 'niggles', 'niggling', 'nigglings', 'niggor', 'niggur', 'niglet', 'nignog', 'nigr', 'nigra', 'nigre', 'nip', 'nipple', 'nipplering', 'nittit', 'nlgger', 'nlggor', 'nofuckingway', 'nook', 'nookey', 'nookie', 'noonan', 'nooner', 'nude', 'nudger', 'nuke', 'nutfucker', 'nymph', 'ontherag', 'oral', 'orga', 'orgasim ', 'orgasm', 'orgies', 'orgy', 'osama', 'paki', 'palesimian', 'palestinian', 'pansies', 'pansy', 'panti', 'panties', 'payo', 'pearlnecklace', 'peck', 'pecker', 'peckerwood', 'pee', 'peehole', 'pee-pee', 'peepshow', 'peepshpw', 'pendy', 'penetration', 'peni5', 'penile', 'penis', 'penises', 'penthouse', 'period', 'perv', 'phonesex', 'phuk', 'phuked', 'phuking', 'phukked', 'phukking', 'phungky', 'phuq', 'pi55', 'picaninny', 'piccaninny', 'pickaninny', 'piker', 'pikey', 'piky', 'pimp', 'pimped', 'pimper', 'pimpjuic', 'pimpjuice', 'pimpsimp', 'pindick', 'piss', 'pissed', 'pisser', 'pisses ', 'pisshead', 'pissin ', 'pissing', 'pissoff ', 'pistol', 'pixie', 'pixy', 'playboy', 'playgirl', 'pocha', 'pocho', 'pocketpool', 'pohm', 'polack', 'pom', 'pommie', 'pommy', 'poo', 'poon', 'poontang', 'poop', 'pooper', 'pooperscooper', 'pooping', 'poorwhitetrash', 'popimp', 'porchmonkey', 'porn', 'pornflick', 'pornking', 'porno', 'pornography', 'pornprincess', 'pot', 'poverty', 'premature', 'pric', 'prick', 'prickhead', 'primetime', 'propaganda', 'pros', 'prostitute', 'protestant', 'pu55i', 'pu55y', 'pube', 'pubic', 'pubiclice', 'pud', 'pudboy', 'pudd', 'puddboy', 'puke', 'puntang', 'purinapricness', 'puss', 'pussie', 'pussies', 'pussy', 'pussycat', 'pussyeater', 'pussyfucker', 'pussylicker', 'pussylips', 'pussylover', 'pussypounder', 'pusy', 'quashie', 'queef', 'queer', 'quickie', 'quim', 'ra8s', 'rabbi', 'racial', 'racist', 'radical', 'radicals', 'raghead', 'randy', 'rape', 'raped', 'raper', 'rapist', 'rearend', 'rearentry', 'rectum', 'redlight', 'redneck', 'reefer', 'reestie', 'refugee', 'reject', 'remains', 'rentafuck', 'republican', 'rere', 'retard', 'retarded', 'ribbed', 'rigger', 'rimjob', 'rimming', 'roach', 'robber', 'roundeye', 'rump', 'russki', 'russkie', 'sadis', 'sadom', 'samckdaddy', 'sandm', 'sandnigger', 'satan', 'scag', 'scallywag', 'scat', 'schlong', 'screw', 'screwyou', 'scrotum', 'scum', 'semen', 'seppo', 'servant', 'sex', 'sexed', 'sexfarm', 'sexhound', 'sexhouse', 'sexing', 'sexkitten', 'sexpot', 'sexslave', 'sextogo', 'sextoy', 'sextoys', 'sexual', 'sexually', 'sexwhore', 'sexy', 'sexymoma', 'sexy-slim', 'shag', 'shaggin', 'shagging', 'shat', 'shav', 'shawtypimp', 'sheeney', 'shhit', 'shinola', 'shit', 'shitcan', 'shitdick', 'shite', 'shiteater', 'shited', 'shitface', 'shitfaced', 'shitfit', 'shitforbrains', 'shitfuck', 'shitfucker', 'shitfull', 'shithapens', 'shithappens', 'shithead', 'shithouse', 'shiting', 'shitlist', 'shitola', 'shitoutofluck', 'shits', 'shitstain', 'shitted', 'shitter', 'shitting', 'shitty ', 'shoot', 'shooting', 'shortfuck', 'showtime', 'sick', 'sissy', 'sixsixsix', 'sixtynine', 'sixtyniner', 'skank', 'skankbitch', 'skankfuck', 'skankwhore', 'skanky', 'skankybitch', 'skankywhore', 'skinflute', 'skum', 'skumbag', 'slant', 'slanteye', 'slapper', 'slaughter', 'slav', 'slave', 'slavedriver', 'sleezebag', 'sleezeball', 'slideitin', 'slime', 'slimeball', 'slimebucket', 'slopehead', 'slopey', 'slopy', 'slut', 'sluts', 'slutt', 'slutting', 'slutty', 'slutwear', 'slutwhore', 'smack', 'smackthemonkey', 'smut', 'snatch', 'snatchpatch', 'snigger', 'sniggered', 'sniggering', 'sniggers', "snigger's", 'sniper', 'snot', 'snowback', 'snownigger', 'sob', 'sodom', 'sodomise', 'sodomite', 'sodomize', 'sodomy', 'sonofabitch', 'sonofbitch', 'sooty', 'sos', 'soviet', 'spaghettibender', 'spaghettinigger', 'spank', 'spankthemonkey', 'sperm', 'spermacide', 'spermbag', 'spermhearder', 'spermherder', 'spic', 'spick', 'spig', 'spigotty', 'spik', 'spit', 'spitter', 'splittail', 'spooge', 'spreadeagle', 'spunk', 'spunky', 'squaw', 'stagg', 'stiffy', 'strapon', 'stringer', 'stripclub', 'stroke', 'stroking', 'stupid', 'stupidfuck', 'stupidfucker', 'suck', 'suckdick', 'sucker', 'suckme', 'suckmyass', 'suckmydick', 'suckmytit', 'suckoff', 'suicide', 'swallow', 'swallower', 'swalow', 'swastika', 'sweetness', 'syphilis', 'taboo', 'taff', 'tampon', 'tang', 'tantra', 'tarbaby', 'tard', 'teat', 'terror', 'terrorist', 'teste', 'testicle', 'testicles', 'thicklips', 'thirdeye', 'thirdleg', 'threesome', 'threeway', 'timbernigger', 'tinkle', 'tit', 'titbitnipply', 'titfuck', 'titfucker', 'titfuckin', 'titjob', 'titlicker', 'titlover', 'tits', 'tittie', 'titties', 'titty', 'tnt', 'toilet', 'tongethruster', 'tongue', 'tonguethrust', 'tonguetramp', 'tortur', 'torture', 'tosser', 'towelhead', 'trailertrash', 'tramp', 'trannie', 'tranny', 'transexual', 'transsexual', 'transvestite', 'triplex', 'trisexual', 'trojan', 'trots', 'tuckahoe', 'tunneloflove', 'turd', 'turnon', 'twat', 'twink', 'twinkie', 'twobitwhore', 'uck', 'uk', 'unfuckable', 'upskirt', 'uptheass', 'upthebutt', 'urinary', 'urinate', 'urine', 'usama', 'uterus', 'vagina', 'vaginal', 'vatican', 'vibr', 'vibrater', 'vibrator', 'vietcong', 'violence', 'virgin', 'virginbreaker', 'vomit', 'vulva', 'wab', 'wank', 'wanker', 'wanking', 'waysted', 'weapon', 'weenie', 'weewee', 'welcher', 'welfare', 'wetb', 'wetback', 'wetspot', 'whacker', 'whash', 'whigger', 'whiskey', 'whiskeydick', 'whiskydick', 'whit', 'whitenigger', 'whites', 'whitetrash', 'whitey', 'whiz', 'whop', 'whore', 'whorefucker', 'whorehouse', 'wigger', 'willie', 'williewanker', 'willy', 'wn', 'wog', "women's", 'wop', 'wtf', 'wuss', 'wuzzie', 'xtc', 'xxx', 'yankee', 'yellowman', 'zigabo', 'zipperhead']
    with open (pathBadWords,'w') as f:
        json.dump(badWords,f)

def checkModel():
    modelName="DialoGPT-medium"
    DialoGPTPath=os.path.join(os.getcwd(),modelName)
    if (not os.path.exists(DialoGPTPath)):
        downloadDialoGPT(DialoGPTPath,modelName)
    else:
        print(modelName,' model exist or there is folder name '+modelName)

    modelName="DialogRPT-human-vs-rand"
    pathRPTUpdown = os.path.join(os.getcwd(),modelName)
    if (not os.path.exists(pathRPTUpdown)):
        downloadDialoGRPT(pathRPTUpdown,modelName)
    else:
        print(modelName,' model exist or there is folder name '+modelName)

def checkFile():
    pathBadWords = os.path.join(os.getcwd(),'badWords.json') 
    if (not os.path.exists(pathBadWords)):
        generateBadWordCorpus(pathBadWords)
    else:
        print('Bad words corpus exist')
if __name__=="__main__":
    packages = ['sentencepiece==0.1.95',
                'jsonschema==3.0.2',
                'torch==1.8.1+cu101',
                'tensorflow==2.1.0',
                'transformers==4.6.0',
                'numpy==1.18.5',
                'Flask==1.1.2',
                'Flask_SocketIO==5.0.3']

    for package in packages :
        install(package)
        
print('Downloading Model..')        
checkModel()
print('Finish Downloading Model')

print('Generating File...')
checkFile()
print('Finish Generating File')
