"""Unit tests for the Grimjaw Intent & Banter Engine."""

import pytest
from services.intent_service import GrimjawIntentEngine


def test_intent_engine_categories():
    engine = GrimjawIntentEngine()

    test_queries = [
        ("What is your favorite color?", ["Bone white", "Black", "Ectoplasm"]),
        ("How old are you?", ["two hundred", "remember", "eighteen twenty-four"]),
        ("What's your name?", ["Grimjaw", "sentinel"]),
        ("Are you friendly?", ["Friendly", "sweet as sugar", "friends"]),
        ("Can I have some candy?", ["Take ONE piece", "safe", "treat", "wrapper"]),
        ("How much candy can I take?", ["Take ONE piece", "one piece", "safe", "treat", "wrapper"]),
        ("Where do you live?", ["crypt", "graveyard", "porch", "tomb", "dirt"]),
        ("How did he die?", ["werewolf", "pirate cannon", "dungeon"]),
        ("Are you dead?", ["Dead? Very!", "deceased", "doornail"]),
        ("What's your problem?", ["trapped on this chilly porch", "no stomach for chocolate", "not grumpy", "cranky"]),
        ("Why are you watching the candy?", ["Sentinel", "watching", "protect", "security", "greedy mortal", "compliance"]),
        ("Do you have any family?", ["cemeteries", "headstone", "spider", "family", "mom"]),
        ("What is your favorite candy?", ["Skittles", "skull", "wrapper", "chocolate"]),
        ("Can you do magic?", ["disappear", "black belt", "superpower", "calcium"]),
        ("Do you like my costume?", ["costume", "flesh", "disguise", "cheekbones", "dressing up"]),
        ("Are you real or plastic?", ["authentic", "bones", "touch me", "hug", "ribs", "selfie", "calcium", "real", "vertebrae", "artificial"]),
        ("You're scary!", ["terror", "Not scared", "dental bill", "Why am I so scary", "Hollow eye sockets", "crypt dust"]),
        ("You are so ugly!", ["cheekbones", "stinky", "costume"]),
        ("Man, you're a jerk.", ["jerk", "begging", "respect", "femur", "sugar-fiend"]),
        ("Tell me a joke", ["guts", "trom-bone", "body shop", "humerus", "joke", "body", "lazy", "ribs", "heart"]),
        ("How do you feel about sand?", ["coarse", "rough", "irritating", "vertebrae", "ribcage"]),
        ("Thank you, bye!", ["dentist", "Farewell", "dark"]),
        ("Do you like pizza?", ["Speak up", "Mumble", "wind", "chatter", "larynx"]),
    ]

    for query, expected_substrings in test_queries:
        engine.reset_conversation()
        response = engine.generate_response(query)
        assert isinstance(response, str)
        assert len(response) > 5
        matched = any(sub.lower() in response.lower() for sub in expected_substrings)
        assert matched, f"Failed for query '{query}': got '{response}'"


def test_intent_engine_prolonged_conversation():
    """Verify that the engine gives proper relevant answers even after 10+ turns."""
    engine = GrimjawIntentEngine()
    questions = [
        "What's your name?",
        "Where do you live?",
        "How old are you?",
        "Are you friendly?",
        "How much candy can I take?",
        "Do you have a brain?",
        "Do you like bone broth?",
        "Can I borrow a rib for my dog?",
        "Tell me a joke",
        "Why don't you have eyes?",
        "Do you know the skeleton dance?",
        "Where do you keep your brain?",
    ]
    for q in questions:
        resp = engine.generate_response(q)
        assert len(resp) > 5
        # Ensure it didn't get overridden by a generic scram on a valid question
        assert "shoo" not in resp.lower() or "shoo" in resp.lower()


def test_intent_engine_third_person_and_expanded_queries():
    """Verify that third-person phrased questions and expanded categories work seamlessly."""
    engine = GrimjawIntentEngine()
    test_cases = [
        ("How did he die?", ["werewolf", "pirate cannon", "dungeon"]),
        ("Why is he watching the candy?", ["Sentinel", "watching", "protect", "security", "greedy mortal", "compliance"]),
        ("Where is his house?", ["crypt", "graveyard", "porch", "tomb", "dirt"]),
        ("What is his name?", ["Grimjaw", "sentinel"]),
        ("What is his favorite song?", ["bass", "ribcage", "Spooky Scary Skeletons", "graveyard"]),
        ("What's your favorite movie?", ["Nightmare Before Christmas", "Jack", "horror", "heroic"]),
        ("Can he do magic?", ["disappear", "black belt", "superpower", "calcium"]),
        ("Does he have a girlfriend?", ["cemeteries", "headstone", "spider", "family", "mom"]),
        ("Why is he so grumpy?", ["trapped on this chilly porch", "no stomach for chocolate", "not grumpy", "cranky"]),
    ]
    for query, expected_substrings in test_cases:
        engine.reset_conversation()
        response = engine.generate_response(query)
        assert isinstance(response, str)
        assert len(response) > 5
        matched = any(sub.lower() in response.lower() for sub in expected_substrings)
        assert matched, f"Failed for query '{query}': got '{response}'"


def test_intent_engine_user_session_queries():
    """Verify specific queries reported from real session transcripts."""
    engine = GrimjawIntentEngine()
    test_cases = [
        ("How much candy can I take?", ["Strictly ONE piece", "Only ONE piece", "One treat only"]),
        ("How long have you been a skeleton?", ["two hundred", "calcium", "Two centuries", "eighteen twenty-four", "buried", "remember"]),
        ("You're a jerk. You're a jerk.", ["jerk", "begging", "bones", "femur", "sugar-fiend", "skull", "respect"]),
        ("What's your problem?", ["trapped on this chilly porch", "no stomach for chocolate", "not grumpy", "cranky"]),
        ("How did you die?", ["werewolf", "pirate cannon", "dungeon"]),
        ("No, they aren't.", ["Deny it", "Argue with a skeleton", "Skeptical"]),
        ("like bad words.", ["Bad words", "calcium deficiency", "clean"]),
        ("Do you get cold without any skin?", ["autumn wind", "sweaters", "skin is", "overrated", "freezing", "goosebumps"]),
        ("Can I borrow a rib for my dog?", ["Borrow a rib", "ears for my soup", "No ribs for rover"]),
        ("How do you see?", ["eyeballs", "supernatural vibrations"]),
        ("Do you miss your skin?", ["Miss my skin", "overrated", "Bare bone"]),
        ("Are you scared of dogs?", ["natural predators", "hound away", "golden retriever", "rattling up a tree"]),
        ("I'm not afraid of you.", ["decorative pumpkin", "Not scared", "supernatural fury"]),
        ("I'm going to steal all the candy.", ["Steal the candy", "calcium donation", "ONE piece", "thief"]),
        ("How would you like a nice greasy pork sandwich?", ["greasy pork sandwich", "ashtray", "Chet", "spinal column"]),
        ("What's your favorite team?", ["Grave Diggers", "undefeated", "defense"]),
        ("Who's your favorite superhero?", ["Ghost Rider", "flaming skull", "Batman", "dark caves"]),
        ("Why did the chicken cross the road?", ["body shop", "barbecue", "graveyard"]),
        ("What's your favorite movie quote?", ["Say hello to my little friend", "I'll be back", "Here's Johnny", "truth", "sand"]),
        ("Give me your favorite Vince and Price quote.", ["Darkness falls", "midnight hour", "funk of forty thousand", "Thriller"]),
        ("What is your favorite Vincent Price quote?", ["Darkness falls", "midnight hour", "funk of forty thousand", "Thriller"]),
        ("Do you like Michael Jackson's Thriller?", ["Darkness falls", "midnight hour", "funk of forty thousand", "Thriller"]),
        ("Are you friends with the mummy?", ["mummy", "tight", "toilet paper", "poker night", "zombies"]),
        ("How do you feel about clerics?", ["holy-symbol", "Turn Undead", "radiant damage"]),
        ("You're stewed, butt-wad!", ["greasy pork sandwich", "ashtray", "Chet", "stewed"]),
        ("Have you tried organizing a holiday swap with Christmas lately?", ["Jack Skellington", "Sandy Claws", "October"]),
        ("How do you feel about He-Man?", ["Skeletor", "Castle Grayskull", "He-Man"]),
        ("Without skin or organs, do you consider yourself naked?", ["Naked", "birthday suit", "Skin is", "Bare bone", "couture"]),
        ("What's your favorite instrument?", ["trom-bone", "organ", "castanets", "xylophone"]),
        ("Who is your favorite skeleton friend?", ["Sans", "Papyrus", "Skeletor", "Jack Skellington", "Indiana Bones"]),
        ("What do you think of Skelator?", ["Skeletor", "Castle Grayskull", "He-Man"]),
        ("Do you live in Castle Greyskull?", ["Castle Grayskull", "He-Man", "crypt"]),
        ("How many experience points do I get if I kill you?", ["XP", "experience points", "loot table", "NPC"]),
        ("Why didn't the skeleton go to the Halloween party?", ["no body to go with", "body"]),
        ("What do you call a skeleton who won't work?", ["Lazy bones", "resting in a crypt"]),
        ("What do skeletons order at a restaurant?", ["Spare ribs"]),
        ("Why couldn't the skeleton rally the team?", ["His heart just wasn't in it", "heart"]),
        ("Who's the most powerful skeleton?", ["most powerful skeleton", "pure calcium superiority", "Grimjaw the Sentinel"]),
        ("Are you friends with the Grim Reaper?", ["Grim Reaper", "Reaper", "scythe"]),
        ("What does Ali G say about skeletons?", ["Why is all skeletons always involved in really spooky stuff", "spooky stuff"]),
        ("What's your favorite Alli G quote?", ["Why is all skeletons always involved in really spooky stuff", "spooky stuff"]),
        ("do you have any questions for C. Everett Koop", ["Why is all skeletons always involved in really spooky stuff", "spooky stuff"]),
        ("Can you sing Spooky Scary Skeletons?", ["Spooky Scary Skeletons", "shivers", "shatter every window"]),
        ("You're a poopyhead.", ["poopy head", "preschool", "digestive tracts", "flesh-bag"]),
        ("They say that you're a dumbass. I didn't say that.", ["jerk", "begging", "femur", "sugar-fiend", "respect"]),
        ("Don't call me a flesh bag.", ["label fits", "skin and guts", "comedic timing"]),
        ("Did you see you in Night of the Living Dead?", ["Night of the Living Dead", "Nightmare Before Christmas", "heroic"]),
        ("Good one.", ["label fits", "skin and guts", "comedic timing"]),
        ("What's your favorite TV show?", ["Tales from the Crypt", "Bones", "Scooby-Doo", "Twilight Zone", "TV show"]),
        ("What's your favorite food?", ["Spare ribs", "Sugar skulls", "bone broth", "chocolate bars"]),
        ("What's your favorite holiday?", ["Halloween", "Day of the Dead"]),
        ("What's your favorite video game?", ["Dark Souls", "MediEvil", "Minecraft", "Undertale"]),
        ("What's your favorite Spice Girls song?", ["wanna be my lover", "zigazig", "friends"]),
        ("What's your favorite hobby?", ["Porch surveillance", "crypt", "scaring", "vertebrae"]),
        ("Are you pretty?", ["great bone structure"]),
        ("Are you handsome?", ["great bone structure"]),
        ("What happened to your hair?", ["Hair", "maintenance-free", "perm"]),
        ("What happened to your eyes?", ["eyeballs", "supernatural vibrations", "vision"]),
        ("Could you lend me 20 bones?", ["Lend you twenty bones", "non-negotiable", "femurs", "bone-broke", "bank"]),
        ("What happened to your heart?", ["heart", "cardiovascular", "organs"]),
        ("Good morning.", ["mortal", "Halloween", "candy", "Grimjaw"]),
        ("How are you doing?", ["bone-tastic", "feeling", "vertebrae", "calcium"]),
        ("How are you feeling?", ["bone-tastic", "feeling", "vertebrae", "calcium"]),
        ("What time is it?", ["witching hour", "Time", "bedtime"]),
        ("Are you happy?", ["Happy", "chocolate", "Bored", "skull grin"]),
        ("Why are you so scary?", ["Why am I so scary", "terror", "dental bill"]),
        ("Hey my brother, can I borrow a copy of your Hey Soul Classics?", ["No, my brother, you have to go buy your own", "buy your own"]),
        ("How much for one rib?", ["Fifty cents", "pour some soda", "change for a hundred"]),
        ("How much is one rib?", ["Fifty cents", "pour some soda", "change for a hundred"]),
        ("I sure am hungry.", ["Fifty cents", "pour some soda", "change for a hundred"]),
        ("i am sure am hungry", ["Fifty cents", "pour some soda", "change for a hundred"]),
        ("trick or treat smell my feet", ["give you something good to eat", "nasal cavity", "manners"]),
        ("smell my feet", ["give you something good to eat", "nasal cavity", "manners"]),
        ("What's your favorite restaurant?", ["Dunne's", "Shepherd’s Pie", "Shepherd's Pie", "sticks to your bones"]),
        ("What is your favorite restaurant", ["Dunne's", "Shepherd’s Pie", "Shepherd's Pie", "sticks to your bones"]),
        ("can you sing Thriller", ["thriller", "darkness falls", "funk of forty thousand"]),
        ("Can you sing Thriller?", ["thriller", "darkness falls", "funk of forty thousand"]),
        ("Can you sing Monster Mash?", ["Monster Mash", "graveyard smash"]),
        ("Can you sing the Adams Family song?", ["Addams Family", "creepy and they're kooky"]),
        ("Can you sing The Time Warp?", ["Time Warp", "jump to the left"]),
        ("Can you read any Edgar Allan Poe?", ["Nevermore", "midnight dreary", "darkness peering"]),
        ("Can you say the raven?", ["Nevermore", "midnight dreary", "darkness peering", "dreaming dreams"]),
        ("Can you read me a poem?", ["Nevermore", "midnight dreary", "darkness peering"]),
        ("can you recite Childe Roland to the Dark Tower came?", ["Childe Roland to the Dark Tower came", "slug-horn"]),
        ("Can you recite child Roland", ["Childe Roland to the Dark Tower came", "slug-horn"]),
        ("Can you recite the Tears in Rain quote from Blade Runner?", ["tears in rain", "tannhäuser gate", "attack ships", "candy"]),
        ("Give me a scary quote", ["something wicked", "abyss", "dark pool", "Dementors", "upon the stair", "Freddy", "census taker", "dead people", "big eyes"]),
        ("Tell me a spooky quote", ["something wicked", "abyss", "dark pool", "Dementors", "upon the stair", "Freddy", "census taker", "dead people", "big eyes"]),
        ("What big eyes you have!", ["all the better to see you with", "see you with"]),
        ("Who is your favorite horror movie character?", ["Cryptkeeper", "Jack Skellington", "Freddy Krueger", "horror character"]),
        ("Is this candy safe to eat?", ["safe", "wrappers", "fresh", "curse", "lurking"]),
        ("What's your favorite number?", ["six seven", "six", "seven"]),
        ("Who's your favorite singer?", ["Rob Zombie", "Zombie", "Dragula"]),
        ("What are you doing?", ["waiting for the Great Pumpkin", "Great Pumpkin"]),
        ("Can you do the magic?", ["magic", "candy disappear", "superpower", "black belt"]),
        ("Can I give you a hug?", ["A hug", "ribs are pointy", "heat warmers", "Freezing"]),
        ("What's your favorite Ali G quote?", ["Why is all skeletons always involved in really spooky stuff?"]),
        ("Give me an Ali G quote", ["Why is all skeletons always involved in really spooky stuff?"]),
        ("What would you ask C. Everett Koop?", ["Why is all skeletons always involved in really spooky stuff?"]),
        ("Do people ever steal candy?", ["greedy mortals", "jaw snaps shut", "empty eye sockets", "grave mistake", "guts"]),
        ("See you later", ["dentist", "Farewell", "dark"]),
        ("Which candy is the best?", ["Skittles", "skull", "wrapper", "chocolate", "gummies"]),
        ("Tell me an Olie G quote.", ["Why is all skeletons always involved in really spooky stuff", "spooky stuff"]),
        ("That's great to hear.", ["Glad to hear it", "little things"]),
        ("Good to hear.", ["Glad to hear it", "little things"]),
        ("How many people have you seen today?", ["Dozens", "sugar-crazed", "flesh-bags", "revolving door"]),
        ("Do you ever travel?", ["anchored", "porch", "iron bolts", "Caribbean cruise"]),
        ("What's your favorite animal?", ["Bats", "spiders", "raven", "dogs", "bones"]),
        ("What's your favorite year?", ["Eighteen twenty-four", "buried", "Nineteen eighty-two", "Thriller"]),
    ]
    for query, expected_substrings in test_cases:
        engine.reset_conversation()
        response = engine.generate_response(query)
        assert isinstance(response, str)
        assert len(response) > 5
        matched = any(sub.lower() in response.lower() for sub in expected_substrings)
        assert matched, f"Failed for query '{query}': got '{response}'"


def test_intent_engine_reset():
    engine = GrimjawIntentEngine()
    engine.generate_response("hello")
    assert engine._turn_count == 1
    engine.reset_conversation()
    assert engine._turn_count == 0


def test_costume_and_quote_specific_matching():
    """Verify that specific character costumes and quotes match their exact responses."""
    engine = GrimjawIntentEngine()

    # Optimus Prime
    resp, intent = engine.classify_and_respond("I'm dressed up as Optimus Prime.")
    assert intent == "costume_optimus_prime"
    assert "Optimus Prime" in resp
    assert "Autobots" in resp

    # Harry Potter
    resp, intent = engine.classify_and_respond("I'm dressed up as Harry Potter.")
    assert intent == "costume_harry_potter"
    assert "Harry Potter" in resp

    # SpongeBob
    resp, intent = engine.classify_and_respond("I'm SpongeBob SquarePants.")
    assert intent == "costume_spongebob"
    assert "SpongeBob" in resp

    # Barbie
    resp, intent = engine.classify_and_respond("I'm dressed as Barbie.")
    assert intent == "costume_barbie"
    assert "Barbie" in resp

    # Spider-Man
    resp, intent = engine.classify_and_respond("I am Spider-Man!")
    assert intent == "costume_spiderman"
    assert "Spider-Man" in resp

    # Say Anything quote (even with "Hey" at the beginning)
    resp, intent = engine.classify_and_respond("Hey my brother, may I borrow a copy of your Hey SO Classics?")
    assert intent == "hey_soul_classics"
    assert "No, my brother" in resp

    # Best friend
    resp, intent = engine.classify_and_respond("Who's your best friend?")
    assert intent == "skeleton_friends"


def test_new_intents_hobby_pets_costume_seinfeld():
    """Verify newly added intents for hobbies, pets, favorite costume, and Seinfeld quotes."""
    engine = GrimjawIntentEngine()

    # So what do you do for hobbies? (ensures 'so what' does not trigger rebuttal)
    resp, intent = engine.classify_and_respond("So what do you do for hobbies?")
    assert intent == "favorite_hobby"
    assert any(w in resp.lower() for w in ["ribcage", "xylophone", "chilling", "hobby", "hobbies", "haunt"])

    # Do you own a pet?
    resp, intent = engine.classify_and_respond("Do you own a pet?")
    assert intent == "pets_and_animals"
    assert any(w in resp.lower() for w in ["raven", "spider", "flesh-hound", "pet"])

    # What is your favorite costume?
    resp, intent = engine.classify_and_respond("What is your favorite costume?")
    assert intent == "favorite_costume"
    assert any(w in resp.lower() for w in ["flesh suit", "skin suit", "bedsheet ghost", "costume"])

    # Tell me a Seinfeld quote
    resp, intent = engine.classify_and_respond("Tell me a Seinfeld quote.")
    assert intent == "seinfeld_quote"
    assert any(w in resp.lower() for w in ["soup", "pretzels", "thirsty"])

