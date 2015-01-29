using System;

namespace Mpdn.PlayerExtensions
{
    namespace Config
    {
        namespace Internal
        {
            public class PlayerExtensions
            {
                // No implementation - we are only using this class as our folder name to pass into ScriptSettingsBase
            }
        }

        public abstract class ScriptSettings<TSettings>
            : ScriptSettingsBase<Internal.PlayerExtensions, TSettings> where TSettings : class, new()
        {
            protected ScriptSettings()
            {
            }

            protected ScriptSettings(TSettings settings)
                : base(settings)
            {
            }
        }
    }
}