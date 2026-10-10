"""Automated test suite verifying that all 100 child statements/questions are recognized."""

import pytest
from services.intent_service import GrimjawIntentEngine

QUESTIONS_AND_STATEMENTS = [
    # Questions
    "Are you wearing a costume or is that just your body?",
    "Do you get cold without any skin on?",
    "Where do you keep your brain?",
    "How do you eat your Halloween candy without a stomach?",
    "Does it hurt when your bones clack together?",
    "Can I borrow a rib for my dog?",
    "Do you have to brush your teeth if you don't have gums?",
    "Why don't you have any eyeballs?",
    "How do you stay standing up without any muscles?",
    "Do you ever get the hiccups?",
    "Are those real bones or plastic?",
    "Where are your clothes?",
    "How do you talk without a tongue?",
    "Do you like bone broth or is that weird for you?",
    "What do you do on days that aren't Halloween?",
    "Does the wind blow right through you?",
    "Do you know the Skeleton Dance song?",
    "Where is your spine attached?",
    "Did you drink all your milk when you were little?",
    "Do your knees squeak when you walk?",
    "Can you tickle yourself?",
    "How do you hold your trick-or-treat bucket with those fingers?",
    "Are you scary or are you friendly?",
    "Do you have a dog skeleton at home?",
    "How do you smile all the time without getting tired?",
    "What happens if you lose a toe bone?",
    "Do you eat dog biscuits?",
    "Can you break apart and put yourself back together?",
    "Do you sleep in a closet or under the bed?",
    "Are you related to the skeleton in my science class?",
    "How tall are you without shoes?",
    "Does it feel weird when it rains on your skull?",
    "Can I take a selfie with you?",
    "Why is your head so smooth?",
    "Do you ever lose your funny bone?",
    "Where do you go when the sun comes up?",
    "Do you have a name?",
    "Is your favorite food rib-eye steak?",
    "Do you like playing hide-and-seek in the dark?",
    "How do you sit down without poking yourself?",
    "Do you get sunburnt in the summer?",
    "Did you lose your skin in an accident?",
    "Can you do a backflip?",
    "Why aren't you wearing shoes?",
    "Do you know where the nearest haunted house is?",
    "Do you want some of my chocolate?",
    "How do you hear me without any ears?",
    "Are you hiding in my garden?",
    "Do you have a heart in there anywhere?",
    "Why are your teeth so big?",
    "Do you get tired of being white and grey?",
    "Can I paint your skull green?",
    "Does soup just spill all over the floor when you eat it?",
    "Are you allowed to go trick-or-treating after midnight?",
    "Do you have any bone jokes?",
    "How do you tie your shoes without thumbs?",
    "Is your favorite holiday Halloween or Day of the Dead?",
    "Can you glow in the dark?",
    "Did you fall off a dinosaur?",
    "Where is your tummy?",
    "Do you wear hats to keep your head warm?",
    "How do you bend your elbows?",
    "Are you scared of dogs?",
    "Can you play the xylophone on your ribs?",
    "Why do you look so hungry?",
    "Do you live in a graveyard?",
    "Can I touch your hand?",
    "Do you like scary movies?",
    "How do you smell things without a nose?",
    "Are you going to eat me?",
    # Statements
    "You look like you need a hug!",
    "My dog would love to chew on your leg!",
    "You have a really cool smile.",
    "You're super tall!",
    "I can see right through your ribs.",
    "You look very skinny.",
    "I like your bone suit!",
    "You look like the guy from The Nightmare Before Christmas!",
    "My mom says I need more calcium, just like you!",
    "You're not very scary, you're funny!",
    "I have bones inside me too!",
    "Your hands are freezing cold!",
    "You need to eat some dinner!",
    "Look, we have the same teeth!",
    "You make a funny noise when you move.",
    "I'm going to dress up as you next year!",
    "You're the best decorated porch on the block!",
    "I'm not afraid of you at all.",
    "Your head looks like a bowling ball!",
    "My x-ray at the doctor looked just like you!",
    "You dropped a finger bone over there!",
    "I think your skull is shining!",
    "You look like a puzzle I built once.",
    "I bet you're really good at tag.",
    "You're missing a shoe!",
    "My pirate costume goes great with your vibe.",
    "You look like you're ready to dance!",
    "Happy Halloween, Mr. Bones!",
    "Don't worry, I won't let any dogs get you!",
    "See you next Halloween!",
]


def test_all_100_questions_and_statements():
    engine = GrimjawIntentEngine()
    fallbacks = set(engine._fallbacks)

    unmatched = []
    for idx, prompt in enumerate(QUESTIONS_AND_STATEMENTS, start=1):
        engine.reset_conversation()
        response = engine.generate_response(prompt)
        assert isinstance(response, str)
        assert len(response) > 5

        if response in fallbacks:
            unmatched.append((idx, prompt, response))

    assert len(unmatched) == 0, f"Unmatched prompts ({len(unmatched)}): {unmatched}"
