import pyfiglet

def generate_intro():
    # Create large ASCII art for "classical_composer"
    large_text = pyfiglet.figlet_format("SFL - Deloitte")
    
    # Add additional intro text
    print(large_text)
    intro_text = f"""\n
    
    Welcome to the Classical Composer pipeline!

    You can incorporate this module into your own projects or use as standalone pipeline. 

    Please see the  README.md for more information on how to use this pipeline/repo.

    """
    print(intro_text)
