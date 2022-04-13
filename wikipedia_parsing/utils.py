from string import punctuation


def filter(option, term):
    option, term = option.lower(), term.lower()
    return (
        "," not in option
        and "film" not in option
        and "band" not in option
        and "album" not in option
        and "musical" not in option
        and "song" not in option
        and "music" not in option
        and "game" not in option
        and "novel" not in option
        and "comic" not in option
        and "name" not in option
        and "disambiguation" not in option
        and "story" not in option
        and "series" not in option
        and "the" not in option
        and "play" not in option
        and "character" not in option
        and "actor" not in option
        and "magazine" not in option
        and "ep" not in option
        and "movie" not in option
        and ":" not in option
        and '"' not in option
        and "river" not in option
        and "lake" not in option
        and "serial" not in option
        and "journal" not in option
        and "channel" not in option
        and "entertainment" not in option
        and "composition" not in option
        and "records" not in option
        and "software" not in option
        and "all pages with titles containing" not in option
        and term in option
    )


def invalid_term(inputString):
    return any(char.isdigit() for char in inputString) or any(p in inputString for p in punctuation)
