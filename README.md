Guideline for LLMs

- Please refer to IDEA.md for the context of this 
- CONTRIBUTING.md has guideline for our preferred coding style
- The solution file is in the root
	- We have Nivara project the core library project, its namespace should be Nivara
	- We have Nivara.Extensions project where we will keep the things built using extensibility
	our library offers, we need to support Apache Arrow, Parquet for IO, and ML.NET Tensors
	compatibility so our library can be used with ML.NET. The needed library references are only
	in this project to force ourselves to always have things that depends on these third party
	libraries only go in this project only
	- We need to cover our work with appropriate unit and integration tests and they all
	will go in tests/Nivara.Tests project
	- samples/Nivara.SampleApp exists to showcase what our library can do and how to properly use it
	- EACH project has README.md in their respective folders for project specific context
- As we build, we should update this README.md, it should follow the industry standards of open source
projects on Github, feel free to remove this existing text
- Check docs/EXAMPLE.md, as we build we should document and reflect our API addition/changes
in there
