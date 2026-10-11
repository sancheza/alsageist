"""Offline Intent and Banter Engine for Grimjaw the Crypt Skeleton.

Provides instant, zero-latency conversational responses to trick-or-treaters
covering over 100 distinct questions, curiosity topics, roasts, jokes, and statements.
"""

import random
import re
from typing import Dict, List, Optional, Set, Tuple


class GrimjawIntentEngine:
    """Classifies user intent and selects rich, non-repeating skeletal dialogue."""

    def __init__(self) -> None:
        """Initialize the intent engine with conversational patterns and responses."""
        self._recently_used: Set[str] = set()
        self._max_recent: int = 25
        self._turn_count: int = 0
        self._asked_costume: bool = False

        self._costume_prompts: List[str] = [
            "Hold on, mortal... who are you supposed to be dressed up as tonight?",
            "Wait a second... what kind of creature are you dressed as?",
            "Before you take another step, tell me: who are you dressed up as?",
        ]

        self._intent_rules: List[Tuple[str, List[str], List[str]]] = [
            # 1a. I'm Gonna Git You Sucka - One Rib
            (
                "one_rib_quote",
                [
                    r"\bhow much (is|for|costs?) (one|a) rib\b",
                    r"\bhow much (is|for|costs?) (the|your)?\s*ribs?\b",
                    r"\b(one rib|order of ribs).*(how much|cost|cents?|price)\b",
                    r"\b(how much|cost|price).*(one rib|a rib)\b",
                    r"\bi'?m gonna git you sucka\b",
                    r"\bi\s*(am\s*)?(sure\s+am|sure|so|really|'m)?\s*(hungry|starving)\b",
                    r"\bi sure am hungry\b",
                    r"\bi am sure am hungry\b",
                ],
                [
                    "Fifty cents! How 'bout I pour some soda in your hand for a nickel? Got change for a hundred?!",
                ],
            ),
            # 1. Dogs: Borrow Rib (Q6)
            (
                "dogs_borrow_rib",
                [
                    r"\bborrow a rib\b",
                    r"\bchew on (your|his) (leg|bone|rib|arm)\b",
                    r"\b(dog|puppy|hound).*\b(chew|bone|rib|leg|arm)\b",
                ],
                [
                    "Borrow a rib for your dog?! How about I borrow one of your ears for my soup? Hands off!",
                    "No ribs for rover! Last year a golden retriever ran off with my left shin bone!",
                ],
            ),
            # 1b. Dogs: Protect (S99)
            (
                "dogs_protect",
                [
                    r"\b(won'?t let|protect|save).*(any )?dogs? get (you|him)\b",
                ],
                [
                    "You'll protect me from dogs? Good, you're officially hired as my Chief Canine Defense Guard.",
                ],
            ),
            # 1c. Dogs: Scared of Dogs (Q63)
            (
                "dogs_scared",
                [
                    r"\b(are you|is he)\s*(scared|afraid|terrified) of (dogs|puppies|hounds)\b",
                    r"\bscared of dogs\b",
                    r"\b(afraid|terrified) of (dogs|puppies|hounds)\b",
                ],
                [
                    "Dogs are the natural predators of the skeletal realm! One bark and I'm rattling up a tree.",
                    "Terrified! Last year a golden retriever ran off with my left shin bone!",
                ],
            ),
            # 1d. Dogs: Biscuits (Q27)
            (
                "dogs_biscuits",
                [
                    r"\bdog biscuits?\b",
                ],
                [
                    "Dog biscuits taste like cardboard and despair. I prefer chocolate... if only I had a stomach!",
                ],
            ),
            # 1e. Dogs: General (Q24, S72)
            (
                "dogs",
                [
                    r"\b(dog|puppy|hound|mutt|canine)\b",
                ],
                [
                    "Keep that hound away from me! Skeletons and dogs are sworn natural enemies.",
                    "Dogs are the natural predators of the skeletal realm! One bark and I'm rattling up a tree.",
                ],
            ),
            # 2a. Body: Cold & Weather (Q2, Q16, S83)
            (
                "cold_and_weather",
                [
                    r"\b(get|are you|is he) cold\b",
                    r"\bwind blow(s)? (right )?through (you|him)\b",
                ],
                [
                    "Cold? The autumn wind whistles right through my ribs like a rusty pipe organ! But hey, I save on sweaters.",
                    "Cold? Freezing! But at least I never get goosebumps—no skin to put them on!",
                ],
            ),
            # 2a2. Body: Sunburnt (Q42)
            (
                "sunburnt",
                [
                    r"\bget sunburnt\b",
                ],
                [
                    "Sunburnt? Skeletons don't burn, mortal! We just get a nice, distinguished antique bone tint.",
                ],
            ),
            # 2b. Body: Naked / Organs
            (
                "naked_and_organs",
                [
                    r"\b(consider yourself|are you) naked\b",
                    r"\bwithout (skin or organs|organs or skin)\b",
                    r"\b(naked|birthday suit)\b",
                ],
                [
                    "Naked? Skeletons aren't naked, we're wearing our birthday suit down to the structural core!",
                    "Without skin or organs, you can't be naked—this is the purest form of minimalist couture!",
                ],
            ),
            # 2b2. Body: Missing Skin & Skin Loss (Q41, S81)
            (
                "missing_skin",
                [
                    r"\b(without (any )?skin|no skin)\b",
                    r"\blose (your|his) skin\b",
                    r"\b(do you|does he) miss (your|his) skin\b",
                    r"\bwhere is (your|his) skin\b",
                ],
                [
                    "Miss my skin? Not at all! Skin is high-maintenance—it wrinkles, it burns, it gets mosquito bites. Bare bone is peak evolution!",
                    "Skin is overrated! It wrinkles, it burns, it gets mosquito bites. Bare bone is peak evolution.",
                ],
            ),
            # 2b. Body: Clothes & Costume (Q1, Q12, S75)
            (
                "clothes_and_costume",
                [
                    r"\b(costume|body)\b.*\b(or is that just (your|his) body|wearing a costume)\b",
                    r"\bwhere are (your|his) clothes\b",
                    r"\bwhy (aren'?t|isn'?t) (you|he) wearing (clothes|a shirt|pants)\b",
                    r"\bbone suit\b",
                ],
                [
                    "This isn't a costume, mortal! This is high-fashion, organic, zero-percent-body-fat living!",
                    "Clothes? Why cover up two hundred years of pristine calcium? Fashion is fleeting, bones are forever.",
                ],
            ),
            # 2c. Body: Bones Inside Me (S81)
            (
                "bones_inside_me",
                [
                    r"\bbones inside me (too)?\b",
                ],
                [
                    "You have bones inside you too? Congratulations, you're just a skeleton waiting to break free!",
                ],
            ),
            # 2d. Body: Heart, Organs & Insides
            (
                "heart_and_organs",
                [
                    r"\b(where is|what happened to|do you have|does he have).*(your|his)?\s*(heart|lungs|liver|kidneys|insides|guts)\b",
                    r"\bhave a heart\b",
                    r"\bwithout a heart\b",
                    r"\bwhat happened to (your|his) heart\b",
                ],
                [
                    "My heart? Decomposed two centuries ago! But that just means nobody can break it.",
                    "No heart, no pulse, no problem! My doctor says my blood pressure is zero over dead. Can't beat that!",
                    "A heart? Skeletons don't need cardiovascular systems—we run on pure spite and sugar fumes!",
                ],
            ),
            # 2e. Body: Anatomy, Tummy, Hungry, Dinner (Q9, Q48, Q60, S76, S83)
            (
                "body_and_skin",
                [
                    r"\bstanding up without (any )?muscles\b",
                    r"\bwhere is (your|his) (tummy|stomach|belly)\b",
                    r"\bsee (right )?through (your|his) ribs\b",
                    r"\blook (very )?skinny\b",
                    r"\beat (some )?dinner\b",
                    r"\blook (so )?hungry\b",
                ],
                [
                    "Standing up without muscles is pure skeletal willpower and stubborn spite! Plus, nobody can ever accuse me of being spineless.",
                    "Where is my tummy? Gone! That means no belly aches, no dieting, and unlimited candy gazing.",
                    "Eat dinner? I'd love to, but calories just slip right through my pelvis onto the floor.",
                ],
            ),
            # 3. Eating, Candy Digestion, Soup, Cannibalism (Q4, Q14, Q38, Q46, Q53, Q70)
            (
                "eating_and_food",
                [
                    r"\beat.*(candy|without a stomach|food)\b",
                    r"\bbone broth\b",
                    r"\brib[\s-]?eye steak\b",
                    r"\bwant some of my (candy|chocolate)\b",
                    r"\bsoup.*(spill|floor)\b",
                    r"\bgoing to eat me\b",
                ],
                [
                    "If I eat candy, it falls right through my ribs onto the porch! Total waste of a Fun Size bar.",
                    "Bone broth? That sounds like cannibalism with extra seasoning! Keep that soup away from me.",
                    "Rib-eye steak? The only rib-eyes around here are the ones staring back at you from my chest!",
                    "Share your chocolate? Hand it over! I'll hold it for safekeeping until next November.",
                    "Eat you? Gross, no! Too much gristle, way too much whining, and not nearly enough calcium.",
                ],
            ),
            # 3a1. What Are You Doing / Great Pumpkin
            (
                "what_are_you_doing",
                [
                    r"\bwhat are (you|he|u) doing\b",
                    r"\bwhat('re| are) you up to\b",
                    r"\bwhat('s| is) he doing\b",
                    r"\bwaiting for (the )?great pumpkin\b",
                    r"\bgreat pumpkin\b",
                ],
                [
                    "I'm waiting for the Great Pumpkin.",
                ],
            ),
            # 3b. Why Watching / Guarding the Candy / Job / Purpose
            (
                "guarding_candy_job",
                [
                    r"\b(why|what).*\b(watch|watching|guard|guarding|protect|protecting|sentinel|security)\b.*\b(candy|bowl|porch|treats?|chocolate|stash)\b",
                    r"\bwhy are (you|he) (watching|guarding|sitting|here|on the porch)\b",
                    r"\bwhat are (you|he) doing (here|on the porch)\b",
                    r"\bwhat('s| is) (your|his) (job|purpose|duty)\b",
                    r"\bwho (put|told|hired) (you|him)\b",
                    r"\bwhy (is he|are you) here\b",
                ],
                [
                    "Why am I watching the candy? Because if I turn around for five seconds, one greedy mortal dumps the entire bowl into their pillowcase!",
                    "I am the appointed Candy Sentinel! I protect the chocolate from thieves, raccoons, and undisciplined toddlers.",
                    "Porch security is serious business! My union contract requires six hours of spooky staring and candy surveillance.",
                    "I'm here to ensure peace, order, and strict one-piece-per-person candy compliance!",
                ],
            ),
            # 3b2. Crowd & Visitors
            (
                "crowd_and_visitors",
                [
                    r"\bhow many (people|mortals|kids|trick-or-treaters|visitors|children)\b",
                    r"\b(seen|met)\s+(many|a lot of|lots of)\s+(people|kids|trick-or-treaters|mortals)\b",
                ],
                [
                    "Dozens of sugar-crazed mortals! Every single one trying to sneak a second piece of candy past these hollow eye sockets.",
                    "Too many flesh-bags to count! My porch has been a revolving door of pint-sized goblins and greedy hands.",
                ],
            ),
            # 3c. How Did He Die / Cause of Death / Origins
            (
                "cause_of_death",
                [
                    r"\b(how|why|when) did (you|he|this guy|mr\.? bones) (die|pass away|perish)\b",
                    r"\bwhat happened to (you|him|your body|his body)\b",
                    r"\bcause of death\b",
                    r"\bhow did (you|he) become a skeleton\b",
                ],
                [
                    "How did I die? Let's just say challenging a werewolf to an arm-wrestling match in eighteen twenty-four was a terrible idea!",
                    "I forgot to eat my vegetables and a runaway pirate cannon did the rest! Two hundred years later, here I am.",
                    "A tragic miscalculation involving a haunted dungeon, a rusty trapdoor, and a cursed amulet!",
                ],
            ),
            # 3c2. Are You Dead / Alive
            (
                "are_you_dead",
                [
                    r"\bare (you|he) (dead|alive)\b",
                    r"\bwere (you|he) alive\b",
                    r"\bis (he|this guy) (dead|alive)\b",
                    r"\bhow long have (you|he) been dead\b",
                ],
                [
                    "Dead? Very! But the sugar rush from watching trick-or-treaters keeps me surprisingly lively.",
                    "I am one hundred percent deceased, but these bones are fully energized for Halloween!",
                    "As dead as a doornail, yet far more conversational than any corpse you've ever met!",
                ],
            ),
            # 3c3. Attitude, Problem, Grumpy
            (
                "attitude_and_problem",
                [
                    r"\bwhat('s| is) (your|his) problem\b",
                    r"\bwhy (are|is) (you|he|this guy) (so )?(mean|grumpy|mad|angry|rude|salty|cranky|sassy)\b",
                    r"\bwhat('s| is) (wrong with|the matter with) (you|him)\b",
                    r"\bwhat is wrong with (you|him)\b",
                    r"\bchill out\b",
                    r"\bcalm down\b",
                    r"\bhave some manners\b",
                    r"\bwhy the attitude\b",
                    r"\bwhat('s| is) (your|his) deal\b",
                ],
                [
                    "My problem? I'm trapped on this chilly porch dealing with sugar-crazed mortals who ask too many questions!",
                    "My problem is having no stomach for chocolate and zero patience for rude mortals!",
                    "I'm not grumpy, I've just been sitting on this porch for two centuries waiting for someone to offer me a good treat!",
                    "You'd be cranky too if you had no eardrums, no skin, and kids demanding candy all night!",
                ],
            ),
            # 3d. Family and Friends
            (
                "family_and_friends",
                [
                    r"\b(do you|does he) have (a |any )?(mom|dad|mother|father|parents|brother|sister|family|wife|husband|girlfriend|boyfriend|friends?)\b",
                    r"\bwho (is|are) (your|his) (family|friends?|parents)\b",
                ],
                [
                    "My family is scattered across seven different historic cemeteries! We only reunite for Halloween and full moons.",
                    "My mother was a lovely marble headstone and my father was a Gothic crypt door. I get my cheekbones from my mom.",
                    "Friends? The ghosts in the attic, the bats in the belfry, and occasionally the spider living in my ribcage.",
                ],
            ),
            # 3e. Specific Favorites
            (
                "favorite_number",
                [
                    r"\b(favorite|best|lucky|like)\b.*\b(number|numeral|digit)\b",
                    r"\bwhat('s| is) (your|his) (favorite )?number\b",
                    r"\bwhat number do (you|he) (like|pick)\b",
                    r"\bfavorite number\b",
                ],
                [
                    "Obviously, it's six seven!",
                ],
            ),
            (
                "favorite_restaurant",
                [
                    r"\b(favorite|best|like)\b.*\b(restaurant|restaraunt|diner|pub|eatery|place to eat)\b",
                    r"\bwhat('s| is) (your|his) favorite (restaurant|restaraunt|diner|pub|eatery|place to eat)\b",
                    r"\bwhere do (you|he) (like to )?(eat|dine|go out)\b",
                    r"\bfavorite (restaurant|restaraunt)\b",
                    r"\bdunne'?s\b",
                ],
                [
                    "Dunne's. Their Shepherd’s Pie really sticks to your bones. HA HA HA",
                ],
            ),
            (
                "favorite_food",
                [
                    r"\b(favorite|best|like)\b.*\b(food|meal|dish|dinner|lunch|breakfast)\b",
                    r"\bwhat (food|meal) do (you|he) (like|eat)\b",
                    r"\bwhat('s| is) (your|his) favorite food\b",
                ],
                [
                    "Favorite food? Spare ribs with extra barbecue sauce! Though it drips right down my spinal column.",
                    "Favorite food? Sugar skulls and calcium-rich bone broth! Zero calories, zero digestion required.",
                    "Favorite food? Fun Size chocolate bars—strictly for gazing at, since I have no stomach!",
                ],
            ),
            (
                "favorite_candy",
                [
                    r"\b(favorite|best|like)\b.*\b(candy|candies|chocolate|treat|snack)\b",
                    r"\b(which|what) candy\b.*\b(best|favorite|good)\b",
                    r"\bcandy\b.*\b(is the best|is your favorite|do you like best)\b",
                    r"\bwhat (candy|chocolate|snack) do (you|he) like\b",
                ],
                [
                    "Favorite candy? Skittles! Or anything with 'skull' on the wrapper. Chocolates melt on my vertebrae.",
                    "Sour worms and skull gummies! Though Fun Size bars are the easiest to admire from afar.",
                ],
            ),
            (
                "favorite_drink",
                [
                    r"\b(favorite|best|like)\b.*\b(drink|beverage|soda|juice|potion)\b",
                    r"\bwhat do (you|he) drink\b",
                ],
                [
                    "Favorite drink? A stiff glass of spirits! Get it? Spirits? ...Tough crowd. Otherwise, ice cold calcium milk—it's udder perfection for bone density!",
                    "A tall glass of bubbling crypt brew with extra calcium ice cubes!",
                ],
            ),
            (
                "favorite_video_game",
                [
                    r"\b(favorite|best|like|play)\b.*\b(video game|game|videogame|gaming|playstation|xbox|nintendo|pc game)\b",
                    r"\bwhat (video game|videogame|game) do (you|he) (play|like)\b",
                    r"\bwhat('s| is) (your|his) favorite (video game|videogame|game)\b",
                ],
                [
                    "Favorite video game? Dark Souls! Skeleton wheels and graveyard bosses are top-tier entertainment.",
                    "Favorite video game? MediEvil! Sir Daniel Fortesque is my skeletal idol.",
                    "Favorite video game? Minecraft! I love watching skeleton archers shoot arrows at clueless players in the dark.",
                    "Favorite video game? Undertale! Sans and Papyrus always make me rattle with laughter.",
                ],
            ),
            (
                "favorite_hobby",
                [
                    r"\b(favorite|best|like)\b.*\b(hobby|hobbies|free time|pastime)\b",
                    r"\bwhat (hobby|hobbies) do (you|he) (have|like|do)\b",
                    r"\bwhat do (you|he) do for (fun|hobbies)\b",
                    r"\b(do you have|got any)\s*(hobbies|a hobby)\b",
                    r"\b(what|do).*(for hobbies|for fun|free time|pastime)\b",
                ],
                [
                    "Favorite hobby? Porch surveillance, scaring sugar-crazed children, and polishing my pristine vertebrae!",
                    "Favorite hobby? Resting in my six-foot crypt, practicing sinister cackles, and organizing bone poker nights.",
                ],
            ),
            (
                "favorite_animal",
                [
                    r"\b(favorite|best|like)\b.*\b(animal|creature|critter)\b",
                    r"\bwhat('s| is) (your|his) favorite (animal|creature|critter)\b",
                    r"\bwhat (animal|creature|critter) do (you|he) like\b",
                ],
                [
                    "Bats and spiders, naturally! They keep the cobwebs tidy and never try to chew on my ribs.",
                    "The raven! Excellent posture, very gothic, and they know how to deliver a dramatic 'Nevermore'.",
                    "Definitely not dogs! Any creature with a natural obsession with bones is my sworn mortal enemy.",
                ],
            ),
            (
                "pets_and_animals",
                [
                    r"\b(do you|does he)\s*(have|own|keep|got).*(a )?(pet|dog|cat|animal|bat|raven|rat|spider)s?\b",
                    r"\b(do you|does he)\s*own (a )?pet\b",
                    r"\bhave any pets\b",
                    r"\bgot a pet\b",
                    r"\bhave a pet\b",
                    r"\bown a pet\b",
                ],
                [
                    "A pet? I tried keeping a dog once, but he kept burying my fibula in the backyard!",
                    "I have a pet raven named Edgar, but he spends all night complaining about someone named Lenore.",
                    "Do the spiders nesting inside my left eye socket count as pets? Because they don't pay rent.",
                ],
            ),
            (
                "favorite_costume",
                [
                    r"\b(favorite|best|coolest)\b.*\b(costume|outfit|getup|disguise)\b",
                    r"\bwhat('s| is) (your|his) favorite (costume|outfit)\b",
                    r"\bwhat (costume|outfit) do (you|he) like best\b",
                ],
                [
                    "My favorite costume? A flesh suit! I used to wear one every single day until it went completely out of style.",
                    "Favorite costume? A bedsheet ghost! It's classic, low-maintenance, and hides how ridiculously skinny I look.",
                ],
            ),
            (
                "favorite_year",
                [
                    r"\b(favorite|best) year\b",
                    r"\bwhat('s| is) (your|his) favorite year\b",
                ],
                [
                    "Eighteen twenty-four, obviously! The year I was buried. Quiet, peaceful, and zero trick-or-treaters pestering me for candy!",
                    "Nineteen eighty-two! Michael Jackson released Thriller, and skeletons finally got the cinematic respect we deserve.",
                ],
            ),
            # 3e1a. Spice Girls & Wannabe
            (
                "spice_girls",
                [
                    r"\bspice girls?\b",
                    r"\bwannabe\b",
                    r"\bzigazig\b",
                ],
                [
                    "If you wanna be my lover, you gotta get with my friends! Make it last forever, friendship never ends! Zigazig ah!",
                    "Yo, I'll tell you what I want, what I really, really want! So tell me what you want, what you really, really want! I wanna zigazig ah!",
                ],
            ),
            # 3e1a1. Thriller & Vincent Price
            (
                "vincent_price_and_thriller",
                [
                    r"\b(vincent price|vince and price|vincent's price)\b",
                    r"\b(thriller|michael jackson)\b",
                    r"\b(sing|play)?\s*thriller\b",
                    r"\bthriller quote\b",
                ],
                [
                    "'Cause this is thriller! Thriller night! And no one's gonna save you from the beast about to strike!",
                    "Darkness falls across the land, the midnight hour is close at hand! Creatures crawl in search of blood, to terrorize your neighborhood! Ha ha ha ha ha!",
                    "The foulest stench is in the air, the funk of forty thousand years! And grizzly ghouls from every tomb are closing in to seal your doom!",
                    "And though you fight to stay alive, your body starts to shiver... For no mere mortal can resist the evil of the Thriller!",
                ],
            ),
            # 3e1a2. Monster Mash
            (
                "monster_mash",
                [
                    r"\bmonster mash\b",
                    r"\b(sing|play)?\s*monster mash\b",
                ],
                [
                    "I was working in the lab late one night, when my eyes beheld an eerie sight! For my monster from his slab began to rise, and suddenly to my surprise... He did the Monster Mash! It was a graveyard smash!",
                    "They did the mash, they did the Monster Mash! The Monster Mash, it was a graveyard smash!",
                ],
            ),
            # 3e1a3. The Addams Family
            (
                "addams_family",
                [
                    r"\b(adams|addams) family\b",
                    r"\b(sing|play)?\s*(the )?(adams|addams) family( song)?\b",
                ],
                [
                    "They're creepy and they're kooky, mysterious and spooky, they're all together ooky, the Addams Family! Snap snap!",
                ],
            ),
            # 3e1a4. The Time Warp
            (
                "time_warp",
                [
                    r"\btime warp\b",
                    r"\brocky horror\b",
                    r"\b(sing|do) (the )?time warp\b",
                ],
                [
                    "It's just a jump to the left, and then a step to the right! Put your hands on your hips, you bring your knees in tight! But it's the pelvic thrust that really drives you insane! Let's do the Time Warp again!",
                ],
            ),
            # 3e1a5. Edgar Allan Poe / The Raven
            (
                "edgar_allan_poe",
                [
                    r"\b(edgar allan poe|the raven|poe|quoth the raven|nevermore)\b",
                    r"\b(read|say|recite|quote) (the )?raven\b",
                    r"\b(read|recite) (any )?(edgar allan )?poe\b",
                    r"\b(can (you|he) )?(read|recite|say|tell)( me)?( a)? (poem|poetry)\b",
                    r"\b(read|recite|say|tell|know)\b.*\b(poem|poetry)\b",
                ],
                [
                    "Once upon a midnight dreary, while I pondered, weak and weary... Quoth the Raven: Nevermore! Now take ONE piece of candy and bother me nevermore!",
                    "Deep into that darkness peering, long I stood there wondering, fearing, doubting... dreaming dreams no mortal ever dared to dream before!",
                ],
            ),
            # 3e1a6. Childe Roland to the Dark Tower Came
            (
                "childe_roland",
                [
                    r"\b(childe? roland|dark tower came|robert browning)\b",
                    r"\bchild(e)?\s+roland\b",
                    r"\b(recite|read|say|tell|know).*\b(child|childe)\s+roland\b",
                ],
                [
                    "Dauntless the slug-horn to my lips I set, and blew: Childe Roland to the Dark Tower came! And this dark tower strictly permits ONE piece of candy per mortal.",
                ],
            ),
            # 3e1a7. Blade Runner - Tears in Rain
            (
                "blade_runner_tears_in_rain",
                [
                    r"\b(tears in rain|blade runner|roy batty|tannh[aä]user gate|attack ships on fire)\b",
                ],
                [
                    "I've seen things you people wouldn't believe... Attack ships on fire off the shoulder of Orion... C-beams glitter in the dark near the Tannhäuser Gate. All those moments will be lost in time, like tears in rain... Time to take your candy!",
                ],
            ),
            # 3e1a8. Say Anything - Hey Soul Classics
            (
                "hey_soul_classics",
                [
                    r"\b(borrow a copy of|copy of|hey soul classics|hey so classics|h-?so classics|soul classics)\b",
                ],
                [
                    "No, my brother! You have to go buy your own!",
                ],
            ),
            # 3e1a9. Seinfeld Quote
            (
                "seinfeld_quote",
                [
                    r"\bseinfeld\b",
                    r"\bno soup for you\b",
                    r"\bthese pretzels are making me thirsty\b",
                    r"\bsoup nazi\b",
                ],
                [
                    "No soup for you! One piece of candy, and come back one year!",
                    "These pretzels... are making me thirsty! And I don't even have a salivary gland!",
                ],
            ),
            (
                "sing_song",
                [
                    r"\b(can (you|he) )?sing( me)?( a)?( song)?\b",
                    r"\bspooky scary skeletons?\b",
                ],
                [
                    "Spooky Scary Skeletons! Shrieking skulls will shock your soul and seal your doom tonight!",
                    "Spooky Scary Skeletons, turned up loud enough to awaken the neighborhood graveyard!",
                    "Sing? If I start singing, the acoustics in my ribcage will shatter every window on this street!",
                ],
            ),
            (
                "favorite_singer",
                [
                    r"\b(favorite|best|like)\b.*\b(singer|vocalist|musician|band|artist)\b",
                    r"\bwho('s| is) (your|his) (favorite )?(singer|vocalist|musician|band|artist)\b",
                    r"\bwho is (your|his) favorite singer\b",
                    r"\brob zombie\b",
                ],
                [
                    "Rob Zombie, obviously! Dragula is the official anthem of this cemetery.",
                    "Rob Zombie! Pure industrial horror perfection for rattling bones.",
                ],
            ),
            (
                "favorite_song",
                [
                    r"\b(favorite|best|like)\b.*\b(song|music|band|tune|soundtrack)\b",
                    r"\bwhat (song|music) do (you|he) like\b",
                ],
                [
                    "Favorite song? Anything with heavy bass that rattles my ribcage like castanets!",
                    "Favorite song? Spooky Scary Skeletons, turned up loud enough to awaken the neighborhood graveyard!",
                ],
            ),
            (
                "favorite_instrument",
                [
                    r"\b(favorite|best|like|play)\b.*\b(instrument|musical instrument)\b",
                    r"\bwhat (instrument) do (you|he) (play|like)\b",
                ],
                [
                    "The trom-bone, obviously! And the rib-cage xylophone when I'm feeling festive.",
                    "The organ in the crypt chapel, though my knuckles make a wonderful set of castanets!",
                ],
            ),
            # 3e1b2. Ali G & C. Everett Koop
            (
                "ali_g",
                [
                    r"\b(ali[\s-]?g|alli[\s-]?g|aly[\s-]?g|ollie[\s-]?g|olie[\s-]?g|alley[\s-]?g|sacha baron cohen|booyakasha)\b",
                    r"\b(c\.?\s*everett\s*koop|everett\s*koop|c\.?\s*everet\s*koop|c\s+everett\s+koop|c\s+everet\s+koop|koop)\b",
                    r"\bquestions? for (c\.?\s*)?everett koop\b",
                    r"\b(quote|saying|line)\s+(from|by)\s+(ali[\s-]?g|alli[\s-]?g|aly[\s-]?g|ollie[\s-]?g|olie[\s-]?g|alley[\s-]?g)\b",
                ],
                [
                    "Why is all skeletons always involved in really spooky stuff?",
                ],
            ),
            # 3e1b3. Say Anything & Hey Soul Classics
            (
                "say_anything_quote",
                [
                    r"\b(hey soul classics?|soul classics?)\b",
                    r"\b(can i borrow|borrow a copy|have a copy).*(hey soul|soul classics?)\b",
                    r"\b(movie say anything|film say anything|say anything movie|lloyd dobler)\b",
                ],
                [
                    "No, my brother, you have to go buy your own!",
                ],
            ),
            # 3e1b2b. Red Riding Hood - Big Eyes
            (
                "red_riding_hood_eyes",
                [
                    r"\bwhat big eyes you have\b",
                    r"\b(why are|why do you have) (such |such a )?big eyes\b",
                    r"\ball the better to see you\b",
                ],
                [
                    "All the better to see you with, my dear!",
                    "What big eyes you have! ...All the better to see you with, my dear!",
                ],
            ),
            # 3e1b3. Scary Quotes
            (
                "scary_quote",
                [
                    r"\b(scary|spooky|creepy|horror|eerie|haunting|dark)\s+(quote|quotes|saying|line)\b",
                    r"\b(say|tell me|give me|recite|know)\s+(a |any )?(scary|spooky|creepy|horror|eerie|dark)\s+(quote|line|saying)\b",
                    r"\bby the pricking of my thumbs\b",
                    r"\bsomething wicked this way comes\b",
                    r"\bgaze.*(into an |the )?abyss\b",
                    r"\bdark pool at the bottom of our hearts\b",
                    r"\bdementors? (are|will)\b",
                    r"\bupon the stair.*met a man who wasn'?t there\b",
                    r"\bfreddy'?s coming for you\b",
                    r"\ba census taker once tried to test me\b",
                    r"\bfava beans and a nice chianti\b",
                    r"\bi see dead people\b",
                ],
                [
                    "By the pricking of my thumbs, something wicked this way comes.",
                    "If you gaze long enough into an abyss, the abyss will gaze back into you.",
                    "Beware the dark pool at the bottom of our hearts. In its icy, black depths dwell strange and twisted creatures it is best not to disturb.",
                    "Dementors are vicious creatures. They will not distinguish between the one they hunt and the one who gets in their way.",
                    "Yesterday, upon the stair, I met a man who wasn't there! He wasn't there again today, I wish, I wish he'd go away!",
                    "One, two, Freddy’s coming for you. Three, four, better lock your door.",
                    "A census taker once tried to test me. I ate his liver with some fava beans and a nice Chianti.",
                    "I see dead people.",
                    "What big eyes you have! ...All the better to see you with, my dear!",
                ],
            ),
            # 3e1c. Favorite Movie Quote
            (
                "favorite_movie_quote",
                [
                    r"\b(favorite|best|coolest|good)\b.*\b(movie quote|film quote|quote)\b",
                    r"\bwhat('s| is) (your|his) (favorite )?(movie )?quote\b",
                    r"\b(say|tell me|give me|know any)\s+(a |your )?(movie )?quote\b",
                    r"\bmovie quote\b",
                ],
                [
                    "Favorite movie quote? I don't like sand. It's coarse and rough and irritating, and it gets everywhere... especially between my vertebrae!",
                    "Favorite movie quote? Say hello to my little friend! ...It's this spider living in my ribcage.",
                    "I'll be back... next Halloween! Hasta la vista, trick-or-treater!",
                    "Here's Johnny! ...Wait, no, here's Grimjaw!",
                    "You can't handle the truth! ...And you certainly can't handle having more than ONE piece of candy!",
                ],
            ),
            (
                "favorite_tv_show",
                [
                    r"\b(favorite|best|like|enjoy|watch)\b.*\b(tv show|tv series|television show|television series|sitcom|cartoon)\b",
                    r"\b(tv show|tv series|television show|television series)\b.*\b(enjoy|like|watch|favorite)\b",
                    r"\bwhat (tv show|tv series|television show) do (you|he) (like|watch|enjoy)\b",
                    r"\bwhat('s| is) (your|his) (favorite )?(tv show|television show|show)\b",
                ],
                [
                    "Favorite TV show? Tales from the Crypt, naturally! The Cryptkeeper and I used to carpool to scare school.",
                    "Favorite TV show? Bones! Though it has far too much forensic science and not enough spooky rattling.",
                    "Favorite TV show? Scooby-Doo! I always root for the haunted skeletons, but those meddling kids ruin everything.",
                    "Favorite TV show? The Twilight Zone, or anything playing on late-night crypt cable!",
                ],
            ),
            # 3e1c1. Favorite Horror / Movie Character
            (
                "favorite_horror_character",
                [
                    r"\b(favorite|best|like)\b.*\b(horror (movie )?character|movie character|horror icon|horror villain|scary character|horror monster|villain)\b",
                    r"\bwho('s| is) (your|his) (favorite )?(horror|movie)?\s*(character|villain|monster|horror icon)\b",
                    r"\bwho is (your|his) favorite (horror )?character\b",
                ],
                [
                    "Favorite horror character? The Cryptkeeper! Excellent cackle, fantastic bone structure, and impeccable comedic timing.",
                    "Favorite horror character? Jack Skellington! He's the Pumpkin King, though he clearly stole my fashion sense.",
                    "Favorite horror character? Freddy Krueger! Great glove, iconic sweater, though he could really use a better skin care routine!",
                ],
            ),
            (
                "favorite_movie",
                [
                    r"\b(favorite|best|like|enjoy|watch)\b.*\b(movie|movies|film|films|cinema)\b",
                    r"\b(movie|movies|film|films)\b.*\b(enjoy|like|watch|favorite)\b",
                    r"\bwhat (kind of )?(movie|movies|film|films) do (you|he) (like|enjoy|watch)\b",
                    r"\bwhat('s| is) (your|his) (favorite )?(movie|film)\b",
                ],
                [
                    "Favorite movie? The Nightmare Before Christmas! Though Jack stole my fashion style.",
                    "Any horror classic where the skeleton gets the final heroic laugh!",
                ],
            ),
            # 3e1d. Night of the Living Dead
            (
                "night_of_the_living_dead",
                [
                    r"\bnight of the living dead\b",
                ],
                [
                    "Night of the Living Dead? Classic cinema! Though the zombies were far too slow and clumsy compared to a proper skeleton.",
                ],
            ),
            # 3e2. Sand & Star Wars Anakin Quote
            (
                "sand_and_star_wars",
                [
                    r"\b(feel about|think of|opinion on|like|hate)\s+sand\b",
                    r"\babout sand\b",
                    r"\bsand\b",
                    r"\b(anakin|skywalker|star wars|darth vader)\b",
                ],
                [
                    "I don't like sand. It's coarse and rough and irritating, and it gets everywhere... especially between my vertebrae!",
                    "Sand? I hate it! It's coarse and rough and gets into every single joint in my ribcage!",
                ],
            ),
            # 3e3. Greasy Pork Sandwich & Weird Science Chet
            (
                "pork_sandwich",
                [
                    r"\b(greasy pork sandwich|pork sandwich)\b",
                    r"\b(stewed|butt[\s-]?wad|buttwad|chet)\b",
                ],
                [
                    "A greasy pork sandwich?! Served in a dirty ashtray?! No thanks, Chet, grease just clogs up my vertebrae!",
                    "You're stewed, butt-wad! How'd you like a greasy pork sandwich served in a dirty ashtray?!",
                    "Greasy pork sandwich? Sounds delightful, but grease just drips right down my spinal column onto the porch!",
                ],
            ),
            # 3e3b. Mummies & Classic Monsters
            (
                "monsters_and_mummy",
                [
                    r"\b(mummy|mummies|frankenstein|dracula|vampire|vampires|zombie|zombies|ghoul|witch|werewolf)\b",
                    r"\bfriends with (the |a )?(mummy|dracula|frankenstein|vampire|zombie)\b",
                ],
                [
                    "The mummy? Oh, we're tight! He's always borrowing my tombstone polish and complaining about toilet paper prices.",
                    "Frankenstein and the mummy are regular guests at crypt poker night. Dracula always folds early!",
                    "Zombies are terrible conversationalists—always 'brains this' and 'brains that'. Skeletons have far more class!",
                ],
            ),
            # 3e3c. D&D Clerics, Paladins, Turn Undead
            (
                "dnd_clerics_and_paladins",
                [
                    r"\b(cleric|clerics|paladin|paladins|turn undead|holy water|smite|radiant damage)\b",
                    r"\bhow do (you|he) feel about (clerics|paladins)\b",
                ],
                [
                    "Clerics?! Keep those holy-symbol-waving killjoys away from me! Turn Undead is terrible for my bone density.",
                    "One blast of radiant damage and I'm scattered across the driveway! Clerics are strictly barred from this porch!",
                ],
            ),
            # 3e3d. D&D XP & Combat
            (
                "dnd_xp_and_combat",
                [
                    r"\b(experience points|xp|level up|loot drop|boss fight|stat block)\b",
                    r"\bkill (you|him) (for xp|for loot)\b",
                ],
                [
                    "Experience points?! You get zero XP and a permanent porch hex! I'm a named NPC, not a random dungeon encounter!",
                    "Zero XP! But my loot table includes one piece of candy and a curse that makes your teeth chatter forever!",
                ],
            ),
            # 3e3e. Skeletor & He-Man
            (
                "skeletor_and_he_man",
                [
                    r"\b(skeletor|skelator|he[\s-]?man|castle gr[ae]yskull|masters of the universe)\b",
                ],
                [
                    "Skeletor? Ah, my loud-mouthed second cousin! Always shouting at He-Man and plotting to conquer Castle Grayskull. He has zero chill.",
                    "Castle Grayskull? High rent, terrible heating, and too many barbarians with swords. My six-foot crypt is much cozier!",
                    "He-Man? That muscle-bound show-off with the bad haircut? Even my calcium is more intimidating than him!",
                ],
            ),
            # 3e3f. Most Powerful Skeleton
            (
                "most_powerful_skeleton",
                [
                    r"\b(most powerful|strongest|toughest|best) skeleton\b",
                    r"\bwho('s| is) the (most powerful|strongest|toughest) skeleton\b",
                ],
                [
                    "The most powerful skeleton? You're looking at him! Two hundred years of pure calcium superiority. Though Skeletor puts up a loud argument.",
                    "Grimjaw the Sentinel! Who else could sit on a freezing porch all night staring down sugar-crazed mortals?",
                ],
            ),
            # 3e3g. Grim Reaper
            (
                "grim_reaper",
                [
                    r"\b(grim reaper|reaper|death himself)\b",
                    r"\bfriends with (the )?grim reaper\b",
                ],
                [
                    "The Grim Reaper? Old Grim and I go way back! He's always trying to recruit me, but I prefer the freelance porch-guarding lifestyle.",
                    "The Reaper? He carries a scythe, I carry an attitude. We respect each other's boundaries.",
                ],
            ),
            # 3e4. Sports & Teams
            (
                "sports_and_teams",
                [
                    r"\b(favorite|best|like)\b.*\b(team|sport|sports|football|baseball|basketball|soccer|hockey)\b",
                    r"\bwhat (team|sport) do (you|he) (like|root for|support)\b",
                    r"\bwhat('s| is) (your|his) (favorite )?team\b",
                ],
                [
                    "Favorite team? The Grave Diggers, obviously! We have an undefeated record six feet under.",
                    "Any team with a killer defense! Though our running game is dead on arrival—our players literally work themselves down to the bare bone!",
                ],
            ),
            # 3e5. Superheroes
            (
                "superheroes",
                [
                    r"\b(favorite|best|like)\b.*\b(superhero|hero|avenger|batman|superman|spiderman)\b",
                    r"\bwho('s| is) (your|his) (favorite )?superhero\b",
                ],
                [
                    "Favorite superhero? Ghost Rider! A flaming skull on a motorcycle is peak aesthetic.",
                    "Batman! He also hangs out in dark caves and avoids human contact.",
                ],
            ),
            # 3f. Magic & Abilities
            (
                "magic_and_abilities",
                [
                    r"\bcan (you|he) (do (the |any )?magic|cast (a )?spell|fly|walk|run|jump|move|disappear|teleport)\b",
                    r"\b(do you|does he) have (magic|powers|superpowers)\b",
                    r"\b(do|perform) (some |a )?magic( trick)?\b",
                ],
                [
                    "Magic? I can make a bowl of candy disappear right before your eyes! Too bad I have no digestive tract.",
                    "No flying, but I have a black belt in spooky porch intimidation and supernatural staring contests!",
                    "My only superpower is surviving for two centuries on zero calories and pure calcium spite.",
                ],
            ),
            # 4. Brain, Thinking, Heart, Hiccups (Q3, Q10, Q49)
            (
                "internal_organs",
                [
                    r"\bwhere.*(keep|is) (your|his) brain\b",
                    r"\bget (the )?hiccups\b",
                    r"\bhave a heart\b",
                ],
                [
                    "Brain? I left my brain in the nineteenth century! Honestly, it was a complete no-brainer—zero headaches, zero stress, and I never have to study for a history exam.",
                    "No diaphragm, no lungs, no hiccups! Just eternal, uninterrupted sarcasm.",
                    "A heart? Only a dark, hollow cavity where my burning love for trick-or-treaters used to be!",
                ],
            ),
            # 5. Teeth, Gums, Smiling, Dental (Q7, Q25, Q50, S84)
            (
                "teeth_and_smile",
                [
                    r"\bbrush (your|his) teeth\b",
                    r"\bwithout (any )?gums\b",
                    r"\bsmile all the time\b",
                    r"\bteeth so big\b",
                    r"\bsame teeth\b",
                    r"\bcool smile\b",
                ],
                [
                    "Zero cavities for two hundred years! My dentist was so impressed he gave me a plaque... and then buried me with it!",
                    "I'm not smiling because I'm happy, kid—I literally don't have lips to frown with!",
                    "Big teeth? All the better to grin at your ridiculous costume with!",
                    "We have the same teeth? Let me check... nope, yours are covered in sugar, mine are pure graveyard enamel!",
                ],
            ),
            # 6a. Senses: Eyes & Seeing (Q8)
            (
                "senses_eyes",
                [
                    r"\bhow (do|can) (you|he) (see|look)\b",
                    r"\b(why don'?t you have|why doesn'?t he have|where are (your|his)).*\b(eye|eyeball)s?\b",
                    r"\bsee without (eyes|eyeballs)\b",
                    r"\bwhat happened to (your|his) (eyes?|eyeballs?)\b",
                ],
                [
                    "I don't need eyeballs! I see through the supernatural vibrations of the crypt.",
                    "No eyeballs needed! The dark energy of Halloween guides my skeletal vision.",
                ],
            ),
            # 6b. Senses: Ears & Hearing (Q47)
            (
                "senses_ears",
                [
                    r"\bhow (do|can) (you|he) hear\b",
                    r"\bhear me without (any )?ears\b",
                    r"\bwhere are (your|his) ears\b",
                ],
                [
                    "Hear you? The wind carries your voice straight through the empty ear holes in my skull.",
                    "No eardrums, but auditory vibrations rattle my skull quite nicely!",
                ],
            ),
            # 6c. Senses: Talking & Tongue (Q13)
            (
                "senses_voice",
                [
                    r"\bhow (do|can) (you|he) (talk|speak)\b",
                    r"\btalk without a tongue\b",
                ],
                [
                    "Talking without a tongue is easy when you've mastered the ancient art of cranial acoustics!",
                ],
            ),
            # 6d. Senses: Smelling & Nose (Q69)
            (
                "senses_nose",
                [
                    r"\bhow (do|can) (you|he) smell\b",
                    r"\bsmell (things )?without a nose\b",
                    r"\bwhere is (your|his) nose\b",
                ],
                [
                    "No nose means I never have to smell wet dog, stinky feet, or burnt pumpkin pie!",
                ],
            ),
            # 7. Clacking, Squeaking, Falling Apart, Reassembly, Xylophone, Spine (Q5, Q18, Q20, Q26, Q28, Q64, S85, S91, S93)
            (
                "bone_mechanics",
                [
                    r"\bhurt when.*bones clack\b",
                    r"\bwhere is (your|his) spine\b",
                    r"\bspine attached\b",
                    r"\bknees squeak\b",
                    r"\blose a toe bone\b",
                    r"\bbreak apart and put (yourself|himself)? back together\b",
                    r"\bplay the xylophone on (your|his) ribs\b",
                    r"\bfunny noise when (you|he) move\b",
                    r"\bdropped a finger bone\b",
                    r"\blook like a puzzle\b",
                ],
                [
                    "Clacking doesn't hurt a bit—it's like having castanets built into your knuckles!",
                    "My spine is attached by ancient curses and a whole lot of spooky tension!",
                    "A little WD-40 on the patella usually fixes the squeak, but on chilly nights I rattle like a tin can.",
                    "Lose a toe bone? Pfft, I just replace it with a stale breadstick from the pantry.",
                    "Break apart? If I scatter, good luck putting two hundred and six bones back in the right order!",
                    "Playing xylophone on my ribs tickles, but Bach sounds surprisingly good on my sternum!",
                    "Dropped a finger bone?! Where?! Don't let the dog see it!",
                ],
            ),
            # 8. Real vs Plastic (Q11)
            (
                "reality",
                [
                    r"\breal (bones )?or plastic\b",
                    r"\bare (you|they|those) real\b",
                    r"\bis (he|that) real\b",
                ],
                [
                    "One hundred percent authentic, aged calcium! No cheap blow-molded plastic here.",
                    "Real? Feel the chilling draft radiating off these vertebrae—nothing artificial about me!",
                ],
            ),
            # 8b. Selfies & Photos (Q33)
            (
                "selfie",
                [
                    r"\b(take a )?selfie\b",
                    r"\btake a picture\b",
                    r"\bphoto\b",
                ],
                [
                    "A selfie? Make sure to use the 'no flesh' filter so my cheekbones really pop!",
                    "Smile for the camera! Wait, I'm already smiling... permanently.",
                ],
            ),
            # 8c. Paint Skull (Q52)
            (
                "paint_skull",
                [
                    r"\bpaint (your|his) skull\b",
                    r"\bpaint (you|him)\b",
                ],
                [
                    "Paint my skull green? What is this, St. Patrick's Day? Scram with that paintbrush!",
                    "Touch this pristine bone with wet paint and I'll paint your candy bag invisible!",
                ],
            ),
            # 8d. Hugs & Touching (Q67, S71, S82)
            (
                "touch_and_hug",
                [
                    r"\b(need|give you|want|give him|get) a hug\b",
                    r"\bcan i (give you a )?hug\b",
                    r"\bhug (you|him)\b",
                    r"\btouch ((your|his) hand|you|him)\b",
                    r"\bhands are freezing cold\b",
                    r"\bcan i touch\b",
                ],
                [
                    "A hug? Careful, my ribs are pointy and I don't give refunds for poked eyes!",
                    "Touch my hand? Careful, my phalanges don't have heat warmers installed.",
                    "Freezing cold hands? What did you expect, hot cocoa running through my bone marrow?",
                ],
            ),
            # 9. Sleeping, Daily Routine, Habitat, Graveyard (Q15, Q29, Q36, Q48, Q66)
            (
                "daily_life_and_habitat",
                [
                    r"\bwhere do (you|he) (live|sleep|stay|haunt|hang out)\b",
                    r"\bwhere is (your|his) (home|house|crypt|coffin|tomb|cemetery|grave)\b",
                    r"\bdo (you|he) live (here|in a graveyard|in the ground|underground|in a house)\b",
                    r"\bwhere (are you|is he|did you come|did he come) from\b",
                    r"\bdays that aren'?t halloween\b",
                    r"\bsleep in a closet\b",
                    r"\bunder the bed\b",
                    r"\bwhere do (you|he) go when the sun comes up\b",
                    r"\bhiding in my garden\b",
                    r"\blive in a graveyard\b",
                ],
                [
                    "Six feet under in the graveyard crypt! People are dying to get in, but the neighbors are totally dead silent.",
                    "When I'm not on this porch, I sleep in a velvet-lined crypt catching up on my beauty rest!",
                    "When the sun rises, I retreat under the porch floorboards until the moon comes back.",
                    "A graveyard crypt is my permanent home! Quiet neighbors, zero utility bills, plenty of fresh dirt.",
                    "Hiding in your garden? No, but I might borrow a few marigolds for my tomb.",
                ],
            ),
            (
                "travel_and_vacation",
                [
                    r"\b(do|did|have) (you|he) (ever )?(travel|take a trip|go on vacation|leave the porch)\b",
                    r"\bdo (you|he) travel\b",
                    r"\b(go|ever go) on vacation\b",
                ],
                [
                    "Travel? I'm anchored to this porch by iron bolts! The only trip I take is an occasional tumble when the autumn wind picks up.",
                    "I spent two hundred years six feet under, so this chilly front porch is practically an exotic Caribbean cruise for me!",
                ],
            ),
            # 9b. Science Class Skeleton (Q30)
            (
                "science_class_skeleton",
                [
                    r"\brelated to the skeleton in (my )?science class\b",
                    r"\bscience class skeleton\b",
                ],
                [
                    "Ah, that's my cousin Phil! He got a full-ride scholarship to hang around biology class all day. Talk about a dead-end job with zero backbone!",
                    "Cousin Phil in the science lab! Tell him Grimjaw said to stop slacking and rattle his ribs!",
                ],
            ),
            # 10. Height, Shoes, Smooth Head, Bowling Ball (Q31, Q34, Q40, Q44, Q56, S74, S89, S95)
            (
                "height_and_appearance",
                [
                    r"\bhow tall (are you|is he|r u)\b",
                    r"\b(your|his) head is so smooth\b",
                    r"\bwhy is (your|his) head so smooth\b",
                    r"\bsit down without poking\b",
                    r"\bwhy (aren'?t you|isn'?t he) wearing shoes\b",
                    r"\btie (your|his) shoes\b",
                    r"\bmissing a shoe\b",
                    r"\bsuper tall\b",
                    r"\bbowling ball\b",
                    r"\b(your|his) skull is shining\b",
                ],
                [
                    "Six feet of pure skeletal majesty! And even taller when I stand on my tippy-phalanges.",
                    "Smooth head? That's two centuries of pure calcium polish, kid! No shampoo required.",
                    "Sitting down takes careful geometry—one wrong angle and I poke my own lumbar spine!",
                    "Shoes? Find me a pair of sneakers that accommodate ten bare metatarsals and we'll talk.",
                    "A bowling ball?! Roll my skull down a lane and see what happens to your shins!",
                    "My skull is shining? Thank you, I buffed it with tombstone wax this morning.",
                ],
            ),
            # 10b. Pretty & Handsome & Bone Structure
            (
                "pretty_and_handsome",
                [
                    r"\b(are you|is he)\s*(pretty|handsome|good[\s-]?looking|attractive|cute|gorgeous|beautiful)\b",
                    r"\b(pretty|handsome|good[\s-]?looking)\b",
                ],
                [
                    "Well, I have great bone structure! Drop-dead gorgeous, in fact—people are dying to look like this.",
                ],
            ),
            # 10c. Hair & Baldness
            (
                "hair_and_bald",
                [
                    r"\b(what happened to|where is|where did.*go).*(your|his) hair\b",
                    r"\b(bald|have (any )?hair|lose (your|his) hair)\b",
                ],
                [
                    "Hair? Skeletons don't do bad hair days! Pure polished calcium is completely maintenance-free.",
                    "What happened to my hair? Two hundred years underground will do that to a perm!",
                ],
            ),
            # 10d. Lend 20 Bones (Money Joke)
            (
                "lend_bones_money",
                [
                    r"\b(lend|borrow|give).*\b(\d+|twenty|ten|fifty|some|a few)\s*bones\b",
                    r"\blend me (some )?bones\b",
                ],
                [
                    "Lend you twenty bones?! I need all two hundred and six of them just to stay upright!",
                    "I'm a skeleton, not a bank! Besides, my credit union went bust when the vault turned into a crypt. I'm completely bone-broke!",
                ],
            ),
            # 11. Dancing, Backflips, Jack Skellington, Music (Q17, Q43, S78, S97)
            # 11a. Jack Skellington & Nightmare Before Christmas (S78)
            (
                "jack_skellington",
                [
                    r"\bjack skellington\b",
                    r"\bnightmare before christmas\b",
                    r"\bguy from the nightmare before christmas\b",
                    r"\b(holiday swap|swap with christmas|take over christmas|trade.*with christmas)\b",
                ],
                [
                    "A holiday swap with Christmas?! Look what happened to Jack Skellington—Sandy Claws almost had a fit! I'm sticking strictly to October.",
                    "Jack Skellington? Ha! Jack stole his entire wardrobe and posture from my crypt closet.",
                    "Jack Skellington wishes he had my authentic cheekbones and zero-percent body fat!",
                ],
            ),
            # 11a2. Skeleton Friends & Famous Skeletons
            (
                "skeleton_friends",
                [
                    r"\b(favorite )?skeleton friend(s)?\b",
                    r"\bfriends with other skeletons\b",
                    r"\bother skeleton friends\b",
                    r"\bwho is your (best|favorite) skeleton\b",
                    r"\b(who is|who's|whos) (your|his) (best )?friend(s)?\b",
                    r"\b(your|his) best friend\b",
                ],
                [
                    "My favorite skeleton friend? Sans and Papyrus are great company, but Skeletor is way too noisy. Jack Skellington is cool when he isn't obsessed with snow!",
                    "My best skeleton buddy is Indiana Bones! Always digging up trouble in ancient tombs.",
                ],
            ),
            # 11b. Dancing, Backflips, Grooving (Q17, Q43, S97)
            (
                "dance_and_backflips",
                [
                    r"\bskeleton dance\b",
                    r"\bbackflip\b",
                    r"\bready to dance\b",
                    r"\bcan (you|he) dance\b",
                ],
                [
                    "The Skeleton Dance? Please, that's amateur hour! I invented the graveyard shiver.",
                    "Backflip? If I try a backflip, I'll scatter across this driveway like a dropped box of toothpicks!",
                    "Ready to dance? Hit the spooky music and watch these femurs groove!",
                ],
            ),
            # 12. Milk, Calcium, X-Rays (Q19, S79, S90)
            (
                "milk_and_calcium",
                [
                    r"\bdrink (all (your|his) )?milk\b",
                    r"\bmore calcium\b",
                    r"\bx[\s-]?ray\b",
                ],
                [
                    "Gallons of whole milk! That's why my femur is strong enough to swat away candy thieves.",
                    "Your mom is right! Drink your milk or you'll end up with floppy, wobbly bones like a jellyfish.",
                    "Your doctor's X-ray looked like me? That's because everyone is secretly a skeleton on the inside!",
                ],
            ),
            # 13. Weather, Rain, Glow in Dark, Dinosaurs (Q32, Q58, Q59)
            (
                "weather_and_dino",
                [
                    r"\brains on (your|his) skull\b",
                    r"\bglow in the dark\b",
                    r"\bfall off a dinosaur\b",
                ],
                [
                    "Rain on the skull feels like tap-dancing mice! A bit ticklish, honestly.",
                    "Glow in the dark? Only when I swallow a handful of glow sticks by accident!",
                    "Dinosaur? I'm old, but I wasn't carpooling with the Velociraptors!",
                ],
            ),
            # 14a. Specific Skeleton Riddles & Punchlines
            (
                "halloween_party_joke",
                [
                    r"\b(why didn'?t|why wouldn'?t) the skeleton go to the (halloween )?party\b",
                    r"\bskeleton.*go to the (halloween )?party\b",
                ],
                [
                    "Why didn't the skeleton go to the Halloween party? He had no body to go with!",
                    "He had no body to go with! Classic bone humor.",
                ],
            ),
            (
                "lazy_bones_joke",
                [
                    r"\bwhat do you call a skeleton who (won'?t|doesn'?t) work\b",
                    r"\bskeleton who (won'?t|doesn'?t) work\b",
                    r"\blazy bones\b",
                ],
                [
                    "What do you call a skeleton who won't work? Lazy bones!",
                    "Lazy bones! And believe me, resting in a crypt is hard work.",
                ],
            ),
            (
                "restaurant_spare_ribs_joke",
                [
                    r"\bwhat do skeletons? (order|eat) at a (restaurant|diner)\b",
                    r"\bskeletons? (order|eat) at a restaurant\b",
                    r"\bspare ribs\b",
                ],
                [
                    "What do skeletons order at a restaurant? Spare ribs!",
                    "Spare ribs! Though I always ask for extra barbecue sauce.",
                ],
            ),
            (
                "rally_the_team_joke",
                [
                    r"\b(why couldn'?t|why didn'?t) the skeleton rally the team\b",
                    r"\brally the team\b",
                ],
                [
                    "Why couldn't the skeleton rally the team? His heart just wasn't in it!",
                    "His heart just wasn't in it! Literally!",
                ],
            ),
            (
                "cross_the_road_joke",
                [
                    r"\bwhy did the (chicken|skeleton) cross the road\b",
                    r"\bcross the road\b",
                ],
                [
                    "Why did the skeleton cross the road? To get to the body shop!",
                    "Why did the chicken cross the road? To escape the zombie barbecue on the other side!",
                ],
            ),
            # 14b. Jokes, Funny Bone, Humor (Q35, Q55, S80)
            (
                "jokes_and_humor",
                [
                    r"\b(ever )?(lose|lost) (your|his) funny bone\b",
                    r"\bfunny bone\b",
                    r"\b(tell me a|have any|know any|tell us a) (bone )?joke(s)?\b",
                    r"\bmake me laugh\b",
                    r"\b(you'?re|he'?s) (funny|not very scary)\b",
                ],
                [
                    "Why didn't the skeleton go to the Halloween party? He had no body to go with!",
                    "What do you call a skeleton who won't work? Lazy bones!",
                    "What do skeletons order at a restaurant? Spare ribs!",
                    "Why couldn't the skeleton rally the team? His heart just wasn't in it!",
                    "Why don't skeletons fight each other? Because we don't have the guts!",
                    "What's a skeleton's favorite musical instrument? The trom-bone!",
                    "Why did the skeleton cross the road? To get to the body shop!",
                    "My funny bone is humerus! Get it? Humerus? Ha! Classic anatomical comedy.",
                    "Not scary, just funny? Watch it, or I'll tell you another dad joke until you scream!",
                ],
            ),
            # 15. Games, Hide & Seek, Tag, Movies, Haunted House (Q39, Q45, Q68, S94)
            (
                "games_and_spooky",
                [
                    r"\bhide[\s-]?and[\s-]?seek\b",
                    r"\bhaunted house\b",
                    r"\bscary movies\b",
                    r"\bgood at tag\b",
                ],
                [
                    "I love hide-and-seek! But every time I hide in the closet, people scream that I'm a 'skeleton in the closet' and ruin the entire game.",
                    "Nearest haunted house? You're standing on the porch of it, mortal! Look around you!",
                    "Scary movies? I love them! The best ones always feature handsome skeletons like me in the lead role.",
                    "Good at tag? Unbeatable! One tap with my bony phalange and you're tagged for eternity.",
                ],
            ),
            # 16a. Favorite Holiday
            (
                "favorite_holiday",
                [
                    r"\b(favorite|best|like)\b.*\b(holiday|season)\b",
                    r"\bwhat('s| is) (your|his) favorite holiday\b",
                    r"\bday of the dead\b",
                ],
                [
                    "Halloween, obviously! The only night of the year when mortals willingly offer candy to decorative corpses.",
                    "Halloween and Day of the Dead! Double the candy, double the respect for skulls and bones.",
                ],
            ),
            # 16b. Pirate, Farewell, Mr. Bones, Greetings (Q54, Q57, S96, S98, S100)
            (
                "holidays_and_farewell",
                [
                    r"\bafter midnight\b",
                    r"\bpirate costume\b",
                    r"\bmr\.? bones\b",
                    r"\bsee you next halloween\b",
                    r"\bbest decorated\b",
                    r"\btickle (yourself|himself)\b",
                    r"\bhold (your|his) trick-?or-?treat bucket\b",
                    r"\bhats to keep (your|his) head warm\b",
                    r"\bbend (your|his) elbows\b",
                    r"\b(white and grey|colored white|tired of being white)\b",
                ],
                [
                    "Trick-or-treating after midnight? That's when the real graveyard shift begins!",
                    "Ahoy, matey! Your pirate costume suits my spooky porch aesthetic perfectly.",
                    "Happy Halloween! Take your candy, stay spooky, and watch out for ghouls in the shadows!",
                    "See you next Halloween! I'll be right here on this porch waiting for you!",
                    "Best decorated porch? Finally, a mortal with refined artistic taste! Take an extra treat!",
                    "Tickle myself? Nope, bone on bone just makes a pleasant rhythmic tapping noise.",
                    "Bucket holding is all about thumb leverage and wrist articulation!",
                    "Tired of being white and grey? Never! It goes with every season, every crypt, and every porch!",
                ],
            ),
            # 17. Core Generic Categories (Name, Age, Color, Fear, Candy, Generic Greetings)
            (
                "favorite_color",
                [
                    r"\b(favorite|best|like)\b.*\b(color|colour)\b",
                    r"\bwhat.*color.*do (you|he) like\b",
                ],
                [
                    "Bone white, obviously! Though a nice shade of tombstone grey is a close second.",
                    "Black as the endless void of the crypt, with just a splash of candy-corn orange!",
                    "Ectoplasm green! It brings out the glowing emptiness in my eye sockets.",
                ],
            ),
            (
                "identity",
                [
                    r"\bwhat('s| is) (your|his) name\b",
                    r"\bwho (are you|is he|r u)\b",
                    r"\bwhat (are you|is he) called\b",
                    r"\bintroduce (yourself|himself)\b",
                    r"\b(do you|does he) have a name\b",
                ],
                [
                    "I am Grimjaw, the ancient keeper of the crypt and supreme guardian of this sugar stash!",
                    "They call me Grimjaw! Former porch sleeper, current Halloween candy sentinel.",
                    "Grimjaw is the name, hoarding candy is the game. Remember it, mortal!",
                ],
            ),
            (
                "age_and_death",
                [
                    r"\bhow long.*\b(skeleton|dead|here|around|been a skeleton|haunting|living|awake)\b",
                    r"\bhow old (are you|is he|r u)\b",
                    r"\bwhat.*age\b",
                    r"\bwhen (were you|was he) born\b",
                    r"\bhow long have (you|he) been (dead|here|a skeleton)\b",
                ],
                [
                    "I stopped counting after two hundred years! Let's just say I'm well-aged like fine crypt dust.",
                    "I've been a skeleton for two hundred and two years! Pristine calcium, zero wrinkles.",
                    "Old enough to remember when trick-or-treating involved offering apples instead of complaining about Fun Size bars!",
                    "I was buried in eighteen twenty-four, but I rose again strictly for the free porch snacks.",
                ],
            ),
            (
                "friendliness",
                [
                    r"\bare you (friendly|nice|good|bad|evil|mean|dangerous)\b",
                    r"\bcan we be friends\b",
                    r"\bwill you (hurt|bite|eat|kill) me\b",
                ],
                [
                    "Friendly? I am guarding high-fructose corn syrup in the freezing cold—what do you think?",
                    "I'm as sweet as sugar and twice as dangerous! Take one piece and nobody gets cursed.",
                    "We can be friends, as long as your hands don't grab more than a single piece of candy!",
                ],
            ),
            (
                "how_are_you",
                [
                    r"\bhow (are|r) (you|u|ya)\s*(doing|feeling|today)?\b",
                    r"\bhow('s| is) (it going|everything|life)\b",
                    r"\bhow are (you|u) feeling\b",
                    r"\bhow are (you|u) doing\b",
                ],
                [
                    "How am I doing? Chilly, hollow, and surrounded by sugar-crazed mortals! Other than that, feeling bone-tastic.",
                    "How am I feeling? A little stiff in the lumbar vertebrae, but energized by the sweet scent of chocolate!",
                    "Doing great for someone who's been buried for two centuries! Pristine calcium, zero complaints.",
                ],
            ),
            (
                "smell_my_feet",
                [
                    r"\bsmell (my|your|his|her|their) feet\b",
                    r"\bsmell my feet\b",
                ],
                [
                    "Smell your feet?! I have no nasal cavity! How about I give you something good to eat before your stinky socks dissolve my porch?",
                    "Smell your feet?! Mortals have no manners! Why don't I give you something good to eat instead?",
                    "Smell your feet? Absolutely not! Take a treat and let me give you something good to eat.",
                ],
            ),
            # Costume: K-Pop Demon Hunters
            (
                "costume_kpop_rumi",
                [r"\brumi\b"],
                ["Rumi! Put down the mystical daggers! My ribcage is already well-ventilated, thank you very much!"],
            ),
            (
                "costume_kpop_mira",
                [r"\bmira\b"],
                ["Mira! Don't you dare banish me! Skeletons have had legal residency in this crypt since seventeen-ninety!"],
            ),
            (
                "costume_kpop_zoey",
                [r"\bzoey\b"],
                ["Zoey! If you're going to hunt demons on my porch, at least do it in four-part vocal harmony!"],
            ),
            (
                "costume_kpop_jinu",
                [r"\bjinu\b"],
                ["Jinu! Are you here to drop a chart-topping hit, or just hunt the monsters lurking behind my candy cauldron?"],
            ),
            (
                "costume_kpop_demon_hunters",
                [r"\b(k[- ]?pop|demon hunter|hunt demons?)\b"],
                ["A K-Pop Demon Hunter! Whoa, hold your choreography, pop star! I'm an ancient skeletal sentinel, not a rogue demon from the underworld!"],
            ),
            # Costume: Mario World
            (
                "costume_mario",
                [r"\b(super )?mario\b", r"\bmario brother\b"],
                ["Mario! It's-a you! Sorry, plumber, but your princess is in another crypt down the street!"],
            ),
            (
                "costume_luigi",
                [r"\bluigi\b"],
                ["Luigi! Stop trembling, green bean! I'm just a skeleton, not King Boo! Put the Poltergust away!"],
            ),
            (
                "costume_peach",
                [r"\b(princess )?peach\b"],
                ["Princess Peach! Did Bowser kidnap you again, or are you just out foraging for mini Snickers bars?"],
            ),
            (
                "costume_yoshi",
                [r"\byoshi\b"],
                ["Yoshi! Mlem! Keep that sticky green tongue out of my candy bowl, dinosaur! One piece only!"],
            ),
            # Costume: Bluey
            (
                "costume_bluey",
                [r"\bbluey\b", r"\bheeler\b", r"\bbandit\b", r"\bchilli\b", r"\bingo\b"],
                ["Bluey! For real life?! A blue heeler on my porch! Just don't ask to play Keepy Uppy with my skull!"],
            ),
            # Costume: PAW Patrol
            (
                "costume_chase",
                [r"\bchase\b", r"\bpolice pup\b"],
                ["Chase is on the case! Please, officer, I swear my graveyard permits are up to date! Don't impound my coffin!"],
            ),
            (
                "costume_skye",
                [r"\bskye\b"],
                ["Skye! This pup's gotta fly... straight off my porch before my cape gets caught in your helicopter rotor!"],
            ),
            (
                "costume_marshall",
                [r"\bmarshall\b", r"\bfire pup\b"],
                ["Marshall! Fired up! But please, no water cannons—my bones are already squeaking from the humidity!"],
            ),
            (
                "costume_paw_patrol",
                [r"\bpaw patrol\b"],
                ["PAW Patrol! No job is too big, no pup is too small... but this candy bowl is strictly one per trick-or-treater!"],
            ),
            # Costume: Pokémon
            (
                "costume_eevee",
                [r"\beevee\b"],
                ["Eevee! Look at that fluffy collar! What are you evolving into tonight? A Jolteon, or a Sugar-eon?"],
            ),
            (
                "costume_pikachu",
                [r"\bpikachu\b"],
                ["Pikachu! Pika-pi! I choose YOU... to take one piece of chocolate and not zap my delicate Raspberry Pi!"],
            ),
            # Costume: Toy Story
            (
                "costume_woody",
                [r"\bwoody\b", r"\bsheriff woody\b"],
                [
                    "Wow, you look terrific! Just like Sheriff Woody! Reach for the sky, cowboy, and grab yourself a treat!",
                    "Well howdy, partner! You're my favorite deputy! Now round up ONE piece of candy before Sid shows up!",
                    "You are a toy! You can't fly! ...Wait, I'm a skeleton, I can't even eat chocolate! Take a treat, Woody!",
                    "There's a snake in my boot! Just kidding, partner, no snakes on this porch—just a friendly skeleton guarding the candy!",
                    "Sheriff Woody! Did you leave Buzz Lightyear back at Pizza Planet, or did he fly off to infinity and beyond?",
                ],
            ),
            (
                "costume_buzz",
                [r"\bbuzz( lightyear)?\b"],
                ["Buzz Lightyear! To infinity and beyond! Just make sure your star command trajectory lands you on the sidewalk."],
            ),
            (
                "costume_jessie",
                [r"\bjessie\b"],
                ["Jessie the Yodeling Cowgirl! Yodel-ay-hee-hoo! Skeletons don't do rodeo, so keep your lasso off my femur!"],
            ),
            (
                "costume_toy_story",
                [r"\btoy story\b"],
                ["To infinity and beyond! Mind the skeleton while you grab your treats, toy!"],
            ),
            # Costume: Disney Frozen
            (
                "costume_elsa",
                [r"\belsa\b"],
                ["Elsa! The cold never bothered me anyway! ...Because I have no nerves, no skin, and no blood circulation!"],
            ),
            (
                "costume_anna",
                [r"\banna\b"],
                ["Anna! Do you want to build a snowman? Because with this October draft, my spine is already halfway to permafrost!"],
            ),
            (
                "costume_frozen",
                [r"\bfrozen\b"],
                ["The cold never bothered me anyway! Grab your treat before it freezes to my porch!"],
            ),
            # Costume: Disney Classics
            (
                "costume_moana",
                [r"\bmoana\b"],
                ["Moana! See the line where the sidewalk meets the street? It calls you! Sail your canoe to the next house!"],
            ),
            (
                "costume_mickey",
                [r"\bmickey( mouse)?\b"],
                ["Mickey Mouse! Hot dog! Oh boy! Keep your cartoon white gloves out of my reserve candy stash!"],
            ),
            (
                "costume_minnie",
                [r"\bminnie( mouse)?\b"],
                ["Minnie Mouse! Polka dots on Halloween? Very chic! Just tell Mickey I don't pay rent to the Mouse House!"],
            ),
            (
                "costume_cinderella",
                [r"\bcinderella\b"],
                ["Cinderella! Midnight is approaching! If you leave a glass slipper on my porch, I'm using it as a candy scoop!"],
            ),
            # Costume: TMNT
            (
                "costume_leo",
                [r"\bleonardo\b"],
                ["Leonardo! Katana swords? Mind the edges, Leo—one wrong slash and my ribcage becomes a xylophone!"],
            ),
            (
                "costume_raph",
                [r"\braphael\b"],
                ["Raphael! Easy on the attitude, turtle! You're a little too spicy for a guy dressed in a green half-shell!"],
            ),
            (
                "costume_donnie",
                [r"\bdonatello\b"],
                ["Donatello! You're the brainy one! Can you figure out the algorithmic probability of me giving you two candies? Spoiler: it's zero!"],
            ),
            (
                "costume_mikey",
                [r"\bmichelangelo\b"],
                ["Michelangelo! Cowabunga, dude! Pizza power doesn't work here, but I do have chocolate! Booyakasha!"],
            ),
            (
                "costume_tmnt",
                [r"\b(ninja turtle|tmnt)\b"],
                ["Cowabunga, dude! Keep your ninja weapons off my tombstone!"],
            ),
            # Costume: Superheroes
            (
                "costume_superman",
                [r"\bsuperman\b", r"\bman of steel\b"],
                ["Superman! Faster than a speeding bullet, more powerful than a locomotive... but powerless against my one-candy rule!"],
            ),
            (
                "costume_wonder_woman",
                [r"\bwonder woman\b"],
                ["Wonder Woman! That Lasso of Truth won't work on me! The truth is, I haven't brushed my teeth in three centuries!"],
            ),
            (
                "costume_hulk",
                [r"\b(the )?hulk\b"],
                ["The Hulk! Hulk smash? Please don't smash, big green! I'm already in two hundred and six pieces!"],
            ),
            # Costume: Marvel Heroes
            (
                "costume_spiderman",
                [r"\bspider[- ]?man\b"],
                ["Spider-Man! Is it true that you catch thieves just like flies? Oh, I'll watch out for you, Spider-Man!"],
            ),
            (
                "costume_miles_morales",
                [r"\bmiles morales\b"],
                ["Miles Morales! What's up, danger?! Don't you use your venom-strike on my audio amplifier, web-head!"],
            ),
            (
                "costume_deadpool",
                [r"\bdeadpool\b"],
                ["Deadpool! Finally, a mortal with worse skin than mine! Maximum effort on that costume, Wade!"],
            ),
            (
                "costume_wolverine",
                [r"\bwolverine\b"],
                ["Wolverine! Snikt! Keep those adamantium claws off my collarbones—this isn't an all-you-can-scratch scratching post!"],
            ),
            # Costume: Gaming & Pop Culture
            (
                "costume_sonic",
                [r"\bsonic( the hedgehog)?\b"],
                ["Sonic the Hedgehog! Gotta go fast! Roll right off my porch before you wake the neighborhood canines!"],
            ),
            (
                "costume_minecraft_steve",
                [r"\bminecraft( steve)?\b"],
                ["Minecraft Steve! Put down that diamond pickaxe! You cannot mine calcium blocks in this biome!"],
            ),
            (
                "costume_spongebob",
                [r"\bspongebob( squarepants)?\b"],
                ["SpongeBob SquarePants! Who lives in a pineapple under the sea? Not me, buddy, I live in a crypt under the oak tree!"],
            ),
            (
                "costume_optimus_prime",
                [r"\boptimus( prime)?\b"],
                ["Optimus Prime! Autobots, roll out! Transform and roll down my driveway to the next charging station!"],
            ),
            (
                "costume_baby_yoda",
                [r"\b(baby yoda|grogu)\b"],
                ["Baby Yoda! Grogu! Stop staring at my candy bowl with those giant adorable eyes! The Force cannot steal treats!"],
            ),
            (
                "costume_astronaut",
                [r"\bastronaut\b"],
                ["An astronaut! Is the candy better in zero gravity, or does it just float directly into your helmet?"],
            ),
            (
                "costume_barbie",
                [r"\bbarbie\b"],
                ["Barbie! Hi Barbie! I'd invite you into the Mojo Dojo Casa Crypt, but there's a strict dress code."],
            ),
            (
                "costume_zombie",
                [r"\b(minecraft zombie|zombie)\b"],
                ["Minecraft Zombie! Bruhhhh... Brains? Sorry, pal, look inside my skull—it's like an empty attic in here!"],
            ),
            (
                "costume_harry_potter",
                [r"\bharry potter\b"],
                ["Harry Potter! Ten points to Gryffindor for bravery, minus fifty points for tracking cemetery mud onto my porch!"],
            ),
            (
                "greeting",
                [
                    r"\btrick or treat\b",
                    r"\bhappy halloween\b",
                    r"\bhello\b",
                    r"\bhi\b",
                    r"\bhey\b",
                    r"\bwhat'?s up\b",
                    r"\b(good morning|good afternoon|good evening|good day|howdy|greetings)\b",
                    r"\bboo\b",
                ],
                [
                    "Halt, little mortal! Reach into my bowl if you dare, but watch your fingers—I might just keep them!",
                    "Happy Halloween, flesh-sack! Take one treat before my jaw decides to snap shut!",
                    "Trick or treat, smell my feet—wait, my feet decomposed a century ago. Just take the candy!",
                    "Ah, another candy scavenger arrives! Present yourself to Grimjaw before taking your tribute.",
                ],
            ),
            (
                "candy_theft_inquiry",
                [
                    r"\b(do|does|did|has|have)\s+(people|anyone|anybody|kids|trick-or-treaters)?\s*(ever\s+)?(steal|stolen|rob|swiped?|dump the bowl)\b",
                    r"\b(do|does|has)\s+(anyone|anybody|people)\s+ever\s+try\s+to\s+steal\b",
                    r"\bdo people (ever )?steal\b",
                    r"\bhas anyone (ever )?tried to steal\b",
                    r"\bdo kids (ever )?steal\b",
                    r"\bdoes anyone (ever )?steal\b",
                ],
                [
                    "A few greedy mortals have tried! But the moment they reach for a handful, my jaw snaps shut and they sprint screaming down the sidewalk!",
                    "They try every Halloween! But once these empty eye sockets light up and my ribs rattle, they drop the loot and bolt!",
                    "A few have tried, but stealing from a skeleton is a grave mistake! Besides, none of them have the guts for it... and trust me, I know all about having no guts!",
                ],
            ),
            (
                "candy_theft",
                [
                    r"\b(steal|rob|dump the bowl|take everything|loot)\b",
                    r"\btake (all|the whole bowl)\b",
                ],
                [
                    "Steal the candy?! Try it and see how fast these skeletal fingers lock down the bowl!",
                    "Halt, thief! My union contract strictly forbids unauthorized bulk confectionery removal!",
                    "Touch more than ONE piece and you'll find out why they call it a funny bone — spoiler: it won't be funny!",
                ],
            ),
            (
                "candy_quantity",
                [
                    r"\bhow (much|many).*\b(candy|candies|pieces?|treats?|can i (take|have|get))\b",
                    r"\bhow many (can i|pieces|candies|treats)\b",
                    r"\bcan i (take|have|grab) (two|three|all|the whole bowl|more)\b",
                    r"\bwhat happens if i take (multiple|more|two|three|too many)\b",
                ],
                [
                    "Strictly ONE piece! Touch a second and you're swapping places with me on this freezing porch — see how you like sitting out here all night!",
                    "Only ONE piece per trick-or-treater! I may have no eyes, but I count every single wrapper.",
                    "One treat only! Greed attracts the wrath of the crypt.",
                ],
            ),
            (
                "candy_safety",
                [
                    r"\bis (the|this|that|your)?\s*(candy|treat|chocolate|piece)?\s*(safe|good|poisoned|poisonous|cursed|edible|okay|ok)\b",
                    r"\bis (the|this|that|your)?\s*candy\b.*\b(safe|good|poison|edible|cursed|eat)\b",
                    r"\b(is it|is this|is the candy)\s+(safe|good|poisoned|edible|okay)\b",
                    r"\b(safe|poisoned|poison)\s+to eat\b",
                ],
                [
                    "The candy is perfectly safe, though I can't vouch for whatever is lurking at the bottom of the bowl!",
                    "It's fresh and delicious! As long as you take only ONE piece, you won't suffer the crypt's curse.",
                    "Perfectly safe for mortals! Just check the wrappers and watch out for my bony fingers.",
                ],
            ),
            (
                "candy_request",
                [
                    r"\b(can i|may i|give me|i want|take|get|grab)\b.*\b(candy|treat|chocolate|piece|bowl)\b",
                    r"\bcan i (have|take) (some|one)\b",
                    r"\bwhere is the candy\b",
                ],
                [
                    "Go ahead, grab a treat! Just remember: greed attracts porch curses.",
                    "Help yourself to ONE treat before my bony fingers snap shut!",
                    "Take one piece from the bowl! Touch a second and you have to sit out here and guard the bowl until sunrise.",
                ],
            ),
            (
                "rebuttals_and_banter",
                [
                    r"\b(no (they|you|it|we) aren'?t|no (they|you|it|we) don'?t|you('re| are) lying|liar|nuh-?uh|whatever|who cares)\b",
                    r"\bnot true\b",
                    r"\bare not\b",
                    r"^(well\s+)?so what\??$",
                    r"\bso what if\b",
                    r"\bwhat of it\b",
                ],
                [
                    "Deny it all you want, but the crypt never lies!",
                    "Argue with a skeleton? You really must have nothing better to do on Halloween!",
                    "Skeptical, are we? Keep talking back and I'll hex your candy bag!",
                ],
            ),
            (
                "bad_words_and_language",
                [
                    r"\b(bad words?|curse|swear|profanity|naughty)\b",
                ],
                [
                    "Bad words? The worst word in my vocabulary is 'calcium deficiency'!",
                    "I keep my language as clean as my polished vertebrae!",
                ],
            ),
            (
                "costume",
                [
                    r"\b(like|look at|see)\b.*\b(my costume|my outfit|what i am)\b",
                    r"\bwhat am i\b",
                    r"\bhow do i look\b",
                    r"\bi('m| am) a\b",
                    r"\bdress up\b",
                    r"\bnext year\b",
                ],
                [
                    "Nice costume! Did you find that in a dumpster, or did you pay actual money to look like that?",
                    "Not bad! But you have far too much flesh on your bones to truly pull off a terrifying look.",
                    "Impressive disguise! For a second, I thought the circus had escaped early tonight.",
                    "Dressing up as me next year? Good luck matching these authentic two-hundred-year-old cheekbones!",
                ],
            ),
            # 17b. Time and Clock
            (
                "time_and_clock",
                [
                    r"\bwhat time is it\b",
                    r"\bwhat('s| is) the time\b",
                    r"\bhave the time\b",
                    r"\btell me the time\b",
                    r"\bgot the time\b",
                    r"\bdo you know what time it is\b",
                ],
                [
                    "What time is it? It's the witching hour! Or candy-eating time, whichever you prefer.",
                    "Time? Time has no meaning when you've been sitting on a porch since eighteen twenty-four!",
                    "It's precisely past your bedtime! Grab a treat and scram before the midnight ghouls wake up.",
                ],
            ),
            # 17b2. Weather and Temperature
            (
                "weather_and_temperature",
                [
                    r"\bwhat('s| is) the (temperature|weather|forecast)\b",
                    r"\bhow('s| is) the (weather|temperature)\b",
                    r"\bhow (cold|warm|hot) is it\b",
                    r"\bwhat is it outside\b",
                    r"\bwhat does the weather look like\b",
                    r"\bwhat('s| is) it like outside\b",
                    r"\btell me the (weather|temperature)\b",
                ],
                [
                    "My internal barometer says it's chilly enough to make your teeth chatter and my ribs rattle! Check the sky yourself, mortal.",
                    "The autumn chill whistles straight through my ribs! Perfect weather for rattling around a porch.",
                ],
            ),
            # 17c. Happy / Mood
            (
                "happy_and_mood",
                [
                    r"\b(are you|is he)\s*(happy|sad|excited|bored|lonely|depressed|cheerful|mad)\b",
                ],
                [
                    "Happy? I am guarding a mountain of chocolate under the moonlight—life is good for a bag of bones!",
                    "Happy? Thrilled! Though with a permanent skull grin, it's hard to tell otherwise.",
                    "Bored? Never! Watching sugar-crazed trick-or-treaters is high-class entertainment.",
                ],
            ),
            (
                "fear_not_afraid",
                [
                    r"\b(i'm|i am) not (scared|afraid)\b",
                    r"\byou('re| are) not scary\b",
                    r"\bnot afraid\b",
                    r"\bnot scared\b",
                ],
                [
                    "Not afraid, eh? That's what the last kid said before he turned into that decorative pumpkin over there.",
                    "Not scared? Just wait until my jaw snaps shut with supernatural fury!",
                ],
            ),
            (
                "fear_scared",
                [
                    r"\b(i'm|i am) (scared|afraid|terrified)\b",
                    r"\byou('re| are) (scary|creepy|frightening|terrifying)\b",
                    r"\bwhy (are you|is he) (so )?(scary|creepy|frightening|terrifying)\b",
                    r"\bwhy so scary\b",
                    r"\byou scared me\b",
                ],
                [
                    "Why am I so scary? Hollow eye sockets, glowing stare, and two hundred years of pure crypt dust!",
                    "Good! A little terror is excellent for your cardiovascular health. Now take your candy and run!",
                    "If you think I'm scary now, wait until you see my dental bill without any teeth!",
                ],
            ),
            (
                "insults_looks",
                [
                    r"\b(you('re| are)?|he('s| is)?)?\s*(so |very )?(ugly|stinky|smelly|trash|gross|dumb-looking|stupid-looking)\b",
                    r"\b(smell bad|smell terrible|look ugly|look terrible|stink)\b",
                    r"\bugly skeleton\b",
                    r"\byou smell\b",
                ],
                [
                    "Ugly? Darling, cheekbones like these take two centuries of decomposition to perfect!",
                    "Stinky? I have no flesh left to rot, mortal! I smell like crisp autumn breeze and graveyard wax.",
                    "Look who's talking! Your costume looks like it crawled out of an abandoned thrift store.",
                ],
            ),
            (
                "preschool_insults",
                [
                    r"\bpoop(y)?[\s-]*(head|face|brain)\b",
                    r"\bbutt[\s-]*(head|face|wad|brain)\b",
                    r"\bpee[\s-]*(head|face|brain)\b",
                ],
                [
                    "A poopy head?! What devastating preschool wit! Skeletons don't even have digestive tracts, flesh-bag.",
                ],
            ),
            (
                "insults_personality",
                [
                    r"\b(jerk|mean|rude|annoying|bad|worst|liar|monster|creep|bully|scumbag|loser|idiot|clown|stupid|dumb|dumb[\s-]?ass|jack[\s-]?ass|moron|bozo|dummy|dork|nerd)\b",
                    r"\byou('re| are)?.*\b(jerk|clown|bully|loser|mean|rude|annoying|idiot|creep|bozo|dummy|dumb[\s-]?ass)\b",
                    r"\bshut up\b",
                    r"\bstupid skeleton\b",
                    r"\bi hate you\b",
                ],
                [
                    "A jerk?! I am an ancient creature of darkness guarding free sugar—show some respect!",
                    "Call me whatever you want, mortal, but you're the one begging a pile of bones for candy!",
                    "I'd tell you to bite me, but you'd probably chip a tooth on my femur.",
                    "Keep talking, sugar-fiend. At least my skull has an excuse for being empty!",
                ],
            ),
            (
                "compliments_and_banter",
                [
                    r"\b(good one|nice one|that('s| is| was) (so |very )?funny|haha|hahaha|lol|great joke|funny joke)\b",
                    r"\b(you('re| are)|he('s| is)) (very |so )?funny\b",
                    r"\bdon'?t call me (a )?(flesh[\s-]?bag|flesh[\s-]?sack)\b",
                ],
                [
                    "Of course it's good! I've had two hundred years to practice my comedic timing.",
                    "If the label fits, wear it! You've got skin and guts, I don't. Case closed, flesh-bag!",
                ],
            ),
            (
                "polite_acknowledgment",
                [
                    r"\b(that('s| is)|it('s| is))\s+(great|good|nice|wonderful|cool)\s+to\s+hear\b",
                    r"\b(glad|happy)\s+to\s+hear(\s+that|\s+it)?\b",
                    r"\bgood\s+to\s+hear\b",
                ],
                [
                    "Glad to hear it! Now grab your treat and let another mortal have their turn.",
                    "Indeed! Two centuries six feet under gives you a wonderful appreciation for the little things.",
                ],
            ),
            (
                "goodbye",
                [
                    r"\bthank you\b",
                    r"\bthanks\b",
                    r"\bbye\b",
                    r"\bgoodbye\b",
                    r"\bsee you (later|around|soon|tomorrow|next year)\b",
                    r"\bsee ya\b",
                    r"^see you\b",
                    r"\bhave a good night\b",
                ],
                [
                    "Don't thank me, thank the dentist you'll be visiting in November! Happy Halloween, scram!",
                    "Farewell, small mortal! Tell your friends there's candy here, but warn them about the skull guardian!",
                    "Off you go into the dark! Watch out for the ghouls in the bushes!",
                ],
            ),
            # 22b. Random Line / Quote / Song / Content Request
            (
                "random_content",
                [
                    r"\b(say|tell me|give me|share|do)\b.*\b(something random|a random line|a random quote|a random song|anything random|random thing)\b",
                    r"\bsay something\b",
                    r"\bsay anything\b",
                    r"\btell me something\b",
                    r"\btell me anything\b",
                    r"\brandom quote\b",
                    r"\brandom line\b",
                ],
                [],  # Dynamically populated from self._random_content_pool
            ),
        ]

        # Compile a curated pool of quotes, songs, jokes, and lore (excluding costume feedback)
        self._random_content_pool: List[str] = []
        for name, _, responses in self._intent_rules:
            if name.startswith("costume") or name in ("costume", "favorite_costume"):
                continue
            for r in responses:
                if r and r not in self._random_content_pool:
                    self._random_content_pool.append(r)

        self._fallbacks: List[str] = [
            "Speak up, flesh-bag! I have no ears, just empty holes in my skull!",
            "Mumble all you want, the candy bowl remains under my watchful gaze!",
            "Did you say something, or was that just the autumn wind whistling through my ribcage?",
            "Less chatter, more candy grabbing! Other mortals are waiting in line.",
            "Unlike me, you have a larynx. Use it to speak up!",
        ]

        self._escalation_lines: List[str] = [
            "Still standing there? You're going to turn into a skeleton yourself if you don't keep moving!",
            "Shoo! Scram! You've got your candy, now go haunt someone else's porch!",
            "Alright, loiterer, your visiting hours at the crypt are officially over!",
        ]

    def reset_conversation(self) -> None:
        """Reset turn counter and costume inquiry state for a new visitor."""
        self._turn_count = 0
        self._asked_costume = False

    def classify_and_respond(self, user_text: str) -> Tuple[str, str]:
        """Classify user text and return both the response and matched intent name.

        Args:
            user_text: Transcribed input string from the user.

        Returns:
            Tuple of (response_text, matched_intent_name).
        """
        if not user_text or not user_text.strip():
            return random.choice(self._fallbacks), "fallback_empty"

        self._turn_count += 1
        clean_text = re.sub(r"[^\w\s\']", " ", user_text.lower()).strip()
        clean_text = re.sub(r"\s+", " ", clean_text)

        # 1. Check intent rules first to always answer direct questions and costume statements accurately
        for intent_name, patterns, responses in self._intent_rules:
            for pattern in patterns:
                if re.search(pattern, clean_text, re.IGNORECASE):
                    if intent_name == "random_content":
                        chosen = self._pick_unique(self._random_content_pool)
                    else:
                        chosen = self._pick_unique(responses)
                    if intent_name in ("goodbye", "holidays_and_farewell") and any(
                        w in clean_text for w in ["bye", "see you", "thank", "next year"]
                    ):
                        self.reset_conversation()
                    return chosen, intent_name

        # 2. Once a conversation has been initiated (Turn 2), occasionally (1 in 3 chance) ask who they are dressed as
        if self._turn_count >= 2 and not self._asked_costume and random.random() < 0.33:
            self._asked_costume = True
            prompt_line = random.choice(self._costume_prompts)
            return prompt_line, "costume_inquiry"

        # 3. Lingerer escalation for unrecognized phrases on late turns
        if self._turn_count >= 8 and random.random() < 0.6:
            chosen = self._pick_unique(self._escalation_lines)
            self.reset_conversation()
            return chosen, "lingerer_escalation"

        # 4. Fallback for unrecognized phrases
        return self._pick_unique(self._fallbacks), "fallback_unrecognized"

    def generate_response(self, user_text: str) -> str:
        """Classify user text and return the best conversational response.

        Args:
            user_text: Transcribed input string from the user.

        Returns:
            A sarcastic, relevant Grimjaw spoken line.
        """
        reply, _ = self.classify_and_respond(user_text)
        return reply

    def _pick_unique(self, choices: List[str]) -> str:
        """Select a random response that hasn't been used recently.

        Args:
            choices: Candidate response strings.

        Returns:
            Selected response string.
        """
        available = [c for c in choices if c not in self._recently_used]
        if not available:
            available = choices
            self._recently_used.clear()

        chosen = random.choice(available)
        self._recently_used.add(chosen)

        if len(self._recently_used) > self._max_recent:
            self._recently_used.pop()

        return chosen
