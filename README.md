# RKD Visual Search

This project serves as a prototype to demonstrate how the RKD (Netherland's Institute of Art History) can enhance it's visual and textual search on rkd.research.nl and expand it's querying capabilities for art researchers using vector search and inference from publicly available LLMS.

This prototype embeds records into a multivector representation, indexing them into a private cluster on QDrant as a vector store. XML files are converted into a csv before each records is preprocessed, embedded, and uploaded into a private cluster. Multiple search options are defined, each aiming to address potential research questions/ use cases for users of RKD Research. A front-end prototype can be run in a locally run server to test queries.
