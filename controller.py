import traceback
class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def scrape(self, pj_type):
        """
        Save the email
        :param email:
        :return:
        """
        try:
            # scrape the model
            self.model.pj_type = pj_type
            return self.model.scrape()
        except:
            # show an error message
            traceback.print_exc()