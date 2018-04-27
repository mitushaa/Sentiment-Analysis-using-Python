
	
/****** Object:  Table [dbo].[Dictionary]    Script Date: 07/11/2017 13:54:30 ******/	
SET ANSI_NULLS ON	
GO	
	
SET QUOTED_IDENTIFIER ON	
GO	
	
CREATE TABLE [dbo].[Dictionary](	
	[RecordID] [bigint] IDENTITY(1,1) NOT NULL,
	[TokenWord] [nvarchar](200) NULL,
	[ServiceRequestWordFlag] [bit] NULL,
	[ComplaintWordFlag] [bit] NULL,
	[FeedbackWordFlag] [bit] NULL,
	[NegativeWordFlag] [bit] NULL,
	[PositiveWordFlag] [bit] NULL,
	[SpecialWordFlag] [bit] NULL,
	[IndustryFlag] [nvarchar](200) NULL,
	[IsActive] [bit] NULL
) ON [PRIMARY]	
	
GO	
	
ALTER TABLE [dbo].[Dictionary] ADD  DEFAULT ((0)) FOR [ServiceRequestWordFlag]	
GO	
	
ALTER TABLE [dbo].[Dictionary] ADD  DEFAULT ((0)) FOR [ComplaintWordFlag]	
GO	
	
ALTER TABLE [dbo].[Dictionary] ADD  DEFAULT ((0)) FOR [FeedbackWordFlag]	
GO		
	
ALTER TABLE [dbo].[Dictionary] ADD  DEFAULT ((0)) FOR [NegativeWordFlag]	
GO	
	
ALTER TABLE [dbo].[Dictionary] ADD  DEFAULT ((0)) FOR [PositiveWordFlag]	
GO	
	
ALTER TABLE [dbo].[Dictionary] ADD  DEFAULT ((0)) FOR [SpecialWordFlag]	
GO	
	
ALTER TABLE [dbo].[Dictionary] ADD  DEFAULT ((1)) FOR [IsActive]	
GO	
