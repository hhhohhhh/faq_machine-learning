from typing import Dict, List, Text, Any

from rasa.shared.core.events import Event


class Dialogue:
    """A dialogue comprises a list of Turn objects
    一个 对话 dialogue 包括2个属性：
    name： conversation_id/ sender_id 来作为用户的唯一标识
    events: 一系列事件
    """

    def __init__(self, name: Text, events: List["Event"]) -> None:
        """This function initialises the dialogue with the dialogue name and the event
        list."""
        self.name = name
        self.events = events

    def __str__(self) -> Text:
        """This function returns the dialogue and turns."""
        return "Dialogue with name '{}' and turns:\n{}".format(
            self.name, "\n\n".join([f"\t{t}" for t in self.events])
        )

    def as_dict(self) -> Dict:
        """This function returns the dialogue as a dictionary to assist in
        serialization."""
        return {"events": [event.as_dict() for event in self.events], "name": self.name}

    @classmethod
    def from_parameters(cls, parameters: Dict[Text, Any]) -> "Dialogue":
        """Create `Dialogue` from parameters.

        Args:
            parameters: Serialised dialogue, should contain keys 'name' and 'events'.

        Returns:
            Deserialised `Dialogue`.

        """

        return cls(
            parameters.get("name"),
            [Event.from_parameters(evt) for evt in parameters.get("events")],
        )
